import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import math, copy
from einops import rearrange


class ConvGnSilu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=16):
        super(ConvGnSilu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=out_planes // groups, num_channels=out_planes)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.silu(x)
        return x

class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, groups=16, bias=True):
        super(BottConv, self).__init__()
        num_group = out_channels // groups
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.norm_layer = nn.GroupNorm(num_groups=num_group, num_channels=out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        x= self.silu(self.norm_layer(x))
        return x

def dwconv(in_channel, kernel=3, stride=1, p=1):
    return nn.Conv2d(in_channel, in_channel, kernel_size=kernel, stride=stride, padding=p, groups=in_channel, bias=False)

class DWConv(nn.Module):
    def __init__(self, in_channel, kerner=3, stride=1, padding=1):
        super(DWConv, self).__init__()
        self.dwconvblock = ConvGnSilu(in_channel, in_channel, kernel_size=kerner, stride=stride, padding=padding, groups=in_channel)
    def forward(self, x):
        return self.dwconvblock(x)

#Sequeeze-and-Excitation (SE) block
class SEBlock(nn.Module):
    def __init__(self, mode, in_channels, ratio):
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.fc(self.global_pooling(x))
        v = self.sigmoid(v)
        return x * v

# prompt-based learner
class PBL(nn.Module):
    def __init__(self, in_channels):
        super(PBL, self).__init__()
        self.prompt = torch.nn.parameter.Parameter(torch.randn(in_channels, requires_grad=True))
        self.btc = BottConv(in_channels, in_channels, in_channels // 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  #（B,C,H,W）->(B,H,W,C)
        cos_sim = F.normalize(x, dim=-1) @ F.normalize(self.prompt[None, None, ..., None], dim=2)
        mask = cos_sim.clamp(0, 1)  #[0,1]
        out = x * mask  #
        out = out.permute(0, 3, 1, 2)
        out = self.btc(out)
        return out

# multi-scale channel attention
class MSCA(nn.Module):
    def __init__(self,in_channels, mode= "avg", ratio=2):
        super(MSCA, self).__init__()
        self.scale1 = BottConv(in_channels, in_channels, in_channels // ratio, kernel_size=3, padding=1)
        self.scale2 = BottConv(in_channels, in_channels, in_channels // ratio, kernel_size=5, padding=2)
        self.se = SEBlock(mode, in_channels, ratio)
        self.se1 = SEBlock(mode, in_channels, ratio)
        self.se2 = SEBlock(mode, in_channels, ratio)
        self.conv = DWConv(in_channels)

    def forward(self,x):
        x1 = self.se1(self.scale1(x))
        x2 = self.se2(self.scale2(x))
        x3 = self.se(x)
        msout = x+x1+x2+x3
        out = self.conv(msout)
        return out

# Multi-scale Scratch-specfic Learner
class MSS(nn.Module):
    def __init__(self,in_channels, ratio=2):
        super(MSS, self).__init__()
        self.scale1 = BottConv(in_channels, in_channels, in_channels // ratio, kernel_size=3, padding=1)
        self.scale2 = BottConv(in_channels, in_channels, in_channels // ratio, kernel_size=5, padding=2)
        self.pbl = PBL(in_channels)
        self.pbl1 = PBL(in_channels)
        self.pbl2 = PBL(in_channels)
        self.ptc = ConvGnSilu(in_channels * 3, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=16)

    def forward(self,x):
        x1 = self.pbl1(self.scale1(x))
        x2 = self.pbl2(self.scale2(x))
        x3 = self.pbl(x)
        out = self.ptc(torch.cat((x1,x2,x3), dim=1))
        return out






