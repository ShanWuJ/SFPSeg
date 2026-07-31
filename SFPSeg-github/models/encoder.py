from .dymamba import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, model,pretrained_model=None):
        super().__init__()
        if pretrained_model:
            load_pretrained(model, pretrained_model)
        self.backbone = model
        self.stem = model.stem
        self.stages = model.stages

    def forward(self, x):
        
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            norm = getattr(self.backbone, f"norm{i + 1}")
            x = norm(x)
            outs.append(x)

        return outs


def load_pretrained(model, pretrained_model):
    checkpoint = torch.load(pretrained_model, map_location='cpu')

    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print("loaded pretrained-model successfully")
    else:
        print("loaded pretrained-model false!")


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DAMamba_T()
    x=torch.randn([2,3,512,512])
    encoder = Encoder(model,pretrained_model='/data1/home/wushan/pre_weights/DAMamba-T.pth')
    encoder.to(device)
    x = x.to(device)
    outlist = encoder(x)
    for out in outlist:
        print(out.shape)