import torch
import pywt
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DWConv, MSS


def create_2d_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    rec_hi = torch.tensor(w.rec_hi, dtype=type)
    rec_lo = torch.tensor(w.rec_lo, dtype=type)
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters



def wavelet_2d_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)  #
    return x


def inverse_2d_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

# Wavelet Prompt Filter
class WPF(nn.Module):
    def __init__(self, in_channels, kernel_size=5, stride=1, wt_type='db1'):
        super(WPF, self).__init__()

        self.in_channels = in_channels
        self.stride = stride
        self.dilation = 1
        #creat wavelet filter
        self.wt_filter, self.iwt_filter = create_2d_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wavelet_convs = nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False)
        self.mss = MSS(in_channels, in_channels)   #
        self.convT = nn.ConvTranspose2d(in_channels,in_channels,kernel_size=3, stride=2,padding=1,output_padding=1)

    def forward(self, x):
        # x:[B,C,H,W]
        x_shape = x.shape
        if (x_shape[2] % 2 > 0) or (x_shape[3] % 2 > 0):
            x_pads = (0, x_shape[3] % 2, 0, x_shape[2] % 2)
            x = F.pad(x, x_pads)

        # begin wavelet transformer
        xwave = wavelet_2d_transform(x, self.wt_filter)  # [B, C, 4, H/2, W/2]
        shape_x = xwave.shape  # [B, C, 4, H/2, W/2]
        xwave = xwave.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])  #[B, C*4, H/2, W/2]
        xwave = self.wavelet_convs(xwave)
        xwave = xwave.reshape(shape_x)  #[B, C, 4, H/2, W/2]

        # different frequency components
        x_ll = xwave[:, :, 0, :, :].reshape(shape_x[0],-1, shape_x[3], shape_x[4])
        x_lh = xwave[:, :, 1, :, :].reshape(shape_x[0],-1, shape_x[3], shape_x[4])
        x_hl = xwave[:, :, 2, :, :].reshape(shape_x[0],-1, shape_x[3], shape_x[4])
        x_hh = xwave[:, :, 3, :, :].reshape(shape_x[0],-1, shape_x[3], shape_x[4])
        x_af = x_ll+x_lh+x_hl+x_hh  #[B,C,H/2,W/2]  all frequency components are sum
        fe = self.mss(x_af)  # featuren enhancement in wavelet domain using msam
        out = fe + x_ll
        out = self.convT(out)  #upsample  [B,C,H,W]
        return out




