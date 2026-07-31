import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvGnSilu, dwconv, DWConv

# high-order semantic introduced denoise filter
class NoiseFilter(nn.Module):
    def __init__(self, lh_channel):  #
        super(NoiseFilter,self).__init__()
        self.in_channel = lh_channel
        self.mid_channel = lh_channel // 2

        # dimension reduction
        self.conv1x1 = ConvGnSilu(self.in_channel, self.mid_channel, kernel_size=1, padding=0)

        #
        self.fiter = nn.Sequential(
            dwconv(in_channel=self.mid_channel, kernel=3),
            nn.GroupNorm(num_groups=self.mid_channel // 16, num_channels=self.mid_channel),
            nn.SiLU(),
            ConvGnSilu(self.mid_channel, 16, kernel_size=1),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, fl, fh):
        fh = self.conv1x1(fh)
        heat_map = F.interpolate(fh, size=(fl.shape[-1], fl.shape[-1]), mode='bilinear', align_corners=False)
        fiter = nn.Sigmoid()(self.fiter(heat_map))
        fl = fl * fiter

        return fl


#add-upsample fusion
class ADUFusion(nn.Module):
    def __init__(self):
        super(ADUFusion, self).__init__()

    def forward(self, x, y):  #x is high layer feature
        b, c, H, W = y.size()
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        out = x + y
        return out


# Bidirectional feature fusion module
class BFF(nn.Module):
    def __init__(self, l1c, l2c, l3c, l4c, channel=64):
        # l1c, l2c, l3c ,l4c is layer1, layer2, layer3, layer4's output of backbone, respectively
        super(BFF, self).__init__()

        self.Translayer1 = ConvGnSilu(l1c, channel, 1)
        self.Translayer2 = ConvGnSilu(l2c, channel, 1)
        self.Translayer3 = ConvGnSilu(l3c, channel, 1)
        self.Translayer4 = ConvGnSilu(l4c, channel, 1)

        self.dwconvup4 = DWConv(in_channel=channel)
        self.dwconvup3 = DWConv(in_channel=channel)
        self.dwconvup2 = DWConv(in_channel=channel)

        self.dwconvd1 = DWConv(in_channel=channel, kerner=3, stride=2, padding=1)
        self.dwconvd2 = DWConv(in_channel=channel, kerner=3, stride=2, padding=1)
        self.dwconvd3 = DWConv(in_channel=channel, kerner=3, stride=2, padding=1)

        self.adf1 = ADUFusion()
        self.adf2 = ADUFusion()
        self.adf3 = ADUFusion()

        self.dconv1 = ConvGnSilu(2 * channel, channel, 1)
        self.dconv2 = ConvGnSilu(2*channel, channel, 1)
        self.dconv3 = ConvGnSilu(2*channel, channel, 1)


    def forward(self,x1,x2,x3,x4):
        # all channel is 64 change channels
        x1 = self.Translayer1(x1)
        x2 = self.Translayer2(x2)
        x3 = self.Translayer3(x3)
        x4 = self.Translayer4(x4)


        # from top-to down fusion
        d4 = self.dwconvup4(x4)
        d3 = self.dwconvup3(self.adf3(d4, x3))
        d2 = self.dwconvup2(self.adf2(d3,x2))
        out1 = self.adf1(d2, x1)

        #from down to top
        d1 = self.dwconvd1(out1)
        out2 = self.dconv1(torch.cat((d1,d2), dim=1))
        out3 = self.dconv2(torch.cat((self.dwconvd2(out2),d3),dim=1))
        out4 = self.dconv3(torch.cat((self.dwconvd3(out3), d4), dim=1))

        return out1, out2, out3, out4  # 1 2 3 4

# if __name__=="__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     l1c, l2c, l3c, l4c = 80, 160, 320, 512
#     model = DBFF(l1c, l2c, l3c, l4c)
#
#     x1 = torch.randn([2, 80, 128, 128])
#     x2 = torch.randn([2, 160, 64, 64])
#     x3 = torch.randn([2, 320, 32, 32])
#     x4 = torch.randn([2, 512, 16, 16])
#     model.to(device)
#     x1 = x1.to(device)
#     x2 = x2.to(device)
#     x3 = x3.to(device)
#     x4 = x4.to(device)
#     out1, out2, out3, out4 = model(x1, x2,x3,x4)
#     outlist = [out1, out2, out3, out4]
#     for out in outlist:
#         print(out.shape)