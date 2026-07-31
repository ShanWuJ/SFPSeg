import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath

from .dymamba import DASSM, LayerNorm, ResDWC
from .blocks import BottConv, SEBlock, MSS, MSCA
from .wavelet import WPF

# DASS2D
class DASS2D(nn.Module):
    def __init__(
            self,
            dim=64,
            token_mixer= DASSM,
            head_dim=16,
            drop_path=0.1,
    ):
        super().__init__()
        self.pos_embed = ResDWC(dim, 3)
        self.token_mixer = token_mixer(dim, head_dim=head_dim)
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.pos_embed(x)
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = self.norm2(x)
        x = self.act(x)
        return x


#Gated multi-scale local feature enhancement
class GMLFE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels
        mid_channels = in_channels // 2
        self.btc3_1 = BottConv(in_channels, out_channels, mid_channels, kernel_size=3, padding=1)
        self.btc3_2 = BottConv(in_channels, out_channels, mid_channels, kernel_size=3, padding=1)
        self.btc3_3 = BottConv(in_channels, out_channels, mid_channels, kernel_size=3, padding=1)

        self.btc5_1 = BottConv(in_channels, out_channels, mid_channels, kernel_size=5, padding=2)
        self.btc5_2 = BottConv(in_channels, out_channels, mid_channels, kernel_size=5, padding=2)
        self.btc5_3 = BottConv(in_channels, out_channels, mid_channels, kernel_size=5, padding=2)

    def forward(self, x):  #[C,H,W]
        sc1 = self.btc3_3(x) * self.btc3_2(self.btc3_1(x))
        sc2 = self.btc5_3(x) * self.btc5_2(self.btc5_1(x))
        sc3 = x
        mlsc = sc1 + sc2 +sc3
        return mlsc

# Morphology-aware Spatial Prompting module (MSP)
class MSP (nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.glb = DASS2D(dim=in_channels)
        self.msl = GMLFE(in_channels)
        self.satt = MSS(in_channels*2)
        self.catt = MSCA(in_channels*2)
        self.btc = BottConv(in_channels * 4, in_channels, in_channels * 2, kernel_size=3, padding=1)
    def forward(self, x):
        f_glb = self.glb(x)
        f_msl = self.msl(x)
        fusio = torch.cat((f_glb, f_msl), dim=1)
        catt_out = self.catt(fusio)
        satt_out = self.satt(fusio)
        out = self.btc(torch.cat((catt_out, satt_out), dim=1))
        return out

# Gated bottle Conv Feed-Forward Network with SE
class GFFN(nn.Module):
    def __init__(self, inchannel, drop=0.1):
        super().__init__()
        self.btc1 = BottConv(inchannel*2, inchannel*2, inchannel, kernel_size=3, stride=1, padding=1, bias=True)
        self.se = SEBlock(mode='avg', in_channels=inchannel*2, ratio=2)
        self.btc2 = BottConv(inchannel*2, inchannel*2, inchannel, kernel_size=3, stride=1, padding=1, bias=True)
        self.btc3 = BottConv(inchannel*2, inchannel*2, inchannel, kernel_size=3, stride=1, padding=1, bias=True)
        self.dropout = nn.Dropout(p=drop)
        self.btc = BottConv(inchannel*2, inchannel, inchannel, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self,x):  #[C,H,W] C=64
        res = x
        x1 = self.btc2(self.se(self.btc1(x)))
        x2 = self.btc3(x)
        out = res + x1 + x2
        out = self.dropout(out)
        out = self.btc(out)
        return out  #[C,H,W]

class Decoder(nn.Module):
    def __init__(self, inchannel):
        super().__init__()
        # global
        self.msp = MSP(inchannel)
        self.wpf = WPF(in_channels=inchannel)
        self.fbt = GFFN(inchannel)

    def forward(self, x):
        out_msp = self.msp(x)
        out_wpf = self.wpf(x)
        out = self.fbt(torch.cat((out_msp,out_wpf), dim=1))
        return out


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Decoder(64)
#
    x1 = torch.randn([2, 64, 128, 128])
#     # x2 = torch.randn([2, 64, 64, 64])
#     # x3 = torch.randn([2, 64, 32, 32])
#     # x4 = torch.randn([2, 64, 16, 16])
    model.to(device)
    x1 = x1.to(device)
#     # x2 = x2.to(device)
#     # x3 = x3.to(device)
#     # x4 = x4.to(device)
    out = model(x1)
    print(out.shape)
