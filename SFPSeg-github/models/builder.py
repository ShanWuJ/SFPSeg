import torch
import torch.nn as nn
from functools import partial

from engine.logger import get_logger
from .encoder import Encoder
from .decoder import Decoder
from .bff import BFF
from .seghead import SegHead, UpsampleConv

logger = get_logger()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc

class BCE_Dice(nn.Module):
    def __init__(self, cfg=None):
        super(BCE_Dice, self).__init__()
        assert cfg.bce_ration <= 1.0, f"{cfg.bce_ration} must <=1"
        self.bce_ration = cfg.bce_ration
        self.dice_ration = 1-self.bce_ration
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()

    def forward(self, y_pred, y_true):
        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        # return self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice
        return self.bce_ration * bce + self.dice_ration * dice

#构建模型
class EncoderDecoder(nn.Module):     #这里是构建模型
    def __init__(self, cfg=None):
        super(EncoderDecoder, self).__init__()

        #encoder
        assert cfg.backbone == 'T' or cfg.backbone == 'S'or cfg.backbone == 'B', f"Unsupported backbone type:{cfg.backbone}"

        # import backbone and decoder
        if cfg.backbone == 'T':
            logger.info('Using backbone: DAMamba_T')
            self.channels = [80, 160, 320, 512]
            from .dymamba import DAMamba_T as backbone
            model = backbone()
        elif cfg.backbone == 'S':
            logger.info('Using backbone: DAMamba_S')
            self.channels = [96, 192, 384, 512]
            from .dymamba import DAMamba_S as backbone
            model = backbone()
        elif cfg.backbone == 'B':
            logger.info('Using backbone: DAMamba_B')
            self.channels = [112, 224, 448, 640]
            from .dymamba import DAMamba_B as backbone
            model = backbone()

        self.encoder = Encoder(model, cfg.pretrained_model)

        ##############################################################################################
        #denoised bidirectional feature fusion
        l1c, l2c, l3c, l4c = self.channels
        bff_channel = 64
        self.bff = BFF(l1c, l2c, l3c, l4c, bff_channel)
        # self.size = [16, 32,64,128]

        ##############################################################################################
        # decoder
        self.decoder = nn.ModuleList()  # 1 2 3 4
        for _ in self.channels:
            decoder = Decoder(bff_channel)
            self.decoder.append(decoder)

        ##############################################################################################
        #upsampleConv
        #segment head
        scale_factors = [4,8,16,32]
        self.up = nn.ModuleList()
        for i in range(len(scale_factors)):
            up = UpsampleConv(bff_channel, bff_channel, scale_factor=2, groups=16)
            self.up.append(up)

        self.seghead = SegHead(bff_channel, scale_factor=2)  # 1 2 3 4

        self.criterion = BCE_Dice(cfg)


    def encode_decode(self, rgb):
        x1,x2,x3,x4 = self.encoder(rgb)  # encode
        out1, out2, out3, out4 = self.bff(x1,x2,x3,x4) #fusion
        outs = [out4, out3, out2, out1]
        previous_out = 0
        for i in range(len(outs)):
            # print(i)
            input_decoder = previous_out + outs[i].to(rgb.device)
            decode_out = self.decoder[i](input_decoder)
            previous_out = self.up[i](decode_out)
        seg_in =previous_out
        pred_out = self.seghead(seg_in)
        return pred_out

    def forward(self, rgb, label=None):

        pred = self.encode_decode(rgb)
        if label is not None:
            loss = self.criterion(pred, label.float())
            return loss
        else:
            return pred

    def flops(self, shape=(3, 512, 512)):
        from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
        import copy

        '''
        code from
        https://github.com/MzeroMiko/VMamba/blob/main/classification/models/vmamba.py#L4
        '''

        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = (torch.randn((1, *shape), device=next(model.parameters()).device))
        print(len(input))
        for i in input:
            print(i.shape)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=input, supported_ops=supported_ops)

        del model, input
        # return sum(Gflops.values())
        return f"params {params} GFLOPs {sum(Gflops.values())}"


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops