import torch
import torch.nn as nn
from .blocks import BottConv, SEBlock, ConvGnSilu


# Gated bottle Conv Feed-Forward Network with SE
"""
class GFFN(nn.Module):
    def __init__(self, inchannel, drop=0.1):
        super().__init__()
        outchannel = inchannel
        midchannel = inchannel // 2
        self.btc1 = BottConv(inchannel, outchannel, midchannel, kernel_size=3, stride=1, padding=1, bias=True)
        self.se = SEBlock(mode='avg', in_channels=outchannel, ratio=2)
        self.btc2 = BottConv(inchannel, outchannel, midchannel, kernel_size=3, stride=1, padding=1, bias=True)
        self.btc3 = BottConv(inchannel, outchannel, midchannel, kernel_size=3, stride=1, padding=1, bias=True)
        self.dropout = nn.Dropout(p=drop)

        self.btc = BottConv(inchannel, inchannel // 4, inchannel // 8, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self,x):  #[C,H,W] C=64
        res = x
        x1 = self.btc2(self.se(self.btc1(x)))
        x2 = self.btc3(x)
        out = res + x1 + x2
        out = self.dropout(out)
        out = self.btc(out)
        return out  #[C/4,H,W]
"""

class UpsampleConv(nn.Module):
    """增强版本：包含批归一化和ReLU激活"""
    def __init__(self, in_channels, out_channels, scale_factor=2, groups=16):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=True
        )
        self.conv_block = BottConv(in_channels, out_channels, in_channels//2, groups=groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_block(x)
        return x

class SegHead(nn.Module):
    def __init__(self, inchannel, scale_factor=2):
        super(SegHead, self).__init__()

        # self.cgs1 = ConvGnSilu(inchannel, inchannel // 2,  kernel_size=3, stride=1, padding=1, groups=1)
        # self.up1 = UpsampleConv(inchannel // 2, out_channels = inchannel // 4, scale_factor=scale_factor)  #[C/8,H,W]
        # self.cgs2 = ConvGnSilu(inchannel // 4, inchannel // 8, kernel_size=3, stride=1, padding=1, groups=1)
        # # self.up2 = UpsampleConv(inchannel // 8, out_channels=inchannel // 16, scale_factor=scale_factor)  # [4,H,W]
        # self.cgs3 = ConvGnSilu(inchannel // 16, inchannel // 32, kernel_size=3, stride=1, padding=1, groups=1)
        # self.predconv1 = BottConv(inchannel // 32, 1, 1, groups=1, kernel_size=1)
        # self.pred = nn.Conv2d(1, 1, kernel_size=1)

        self.se = SEBlock(mode='avg', in_channels=inchannel, ratio=2)
        self.cgs1 = BottConv(inchannel, inchannel // 2, inchannel // 2, kernel_size=3, stride=1, padding=1, groups=1)
        self.up1 = UpsampleConv(inchannel // 2, out_channels = inchannel // 4, scale_factor=scale_factor)  #[C/8,H,W]
        self.cgs2 = BottConv(inchannel // 4, inchannel // 8, inchannel // 8, kernel_size=3, stride=1, padding=1, groups=1)
        self.cgs3 = BottConv(inchannel // 8, inchannel // 16, inchannel // 16, kernel_size=3, stride=1, padding=1, groups=1)
        self.cgs4 = BottConv(inchannel // 16, inchannel // 32, inchannel // 32, kernel_size=3, stride=1, padding=1, groups=1)
        self.predconv = BottConv(inchannel // 32, 1, 1, groups=1, kernel_size=1)
        self.pred = nn.Conv2d(1, 1, kernel_size=1)


    def forward(self, x):  #[B,C,H,W]
        x = self.se(x)  #[64,H,W]
        x = self.cgs1(x) #[32,H,W]
        x = self.up1(x)  #[16,H_in,W_in]
        x = self.cgs2(x)  # [8,H_in,W_in]
        x = self.cgs3(x)  # [4,H_in,W_in]
        x = self.cgs4(x)  # [4,H_in,W_in]
        pred = self.predconv(x)  #[1,H_in,W_in]
        pred = self.pred(pred)
        return pred

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegHead(64)
    x1 = torch.randn([2, 64, 256, 256])  #4
    x1 = x1.to(device)
    model.to(device)
    out = model(x1)
    print(out.shape)