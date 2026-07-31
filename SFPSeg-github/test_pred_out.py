import os
import time
import argparse

import pandas
from tqdm import tqdm
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from dataloader.scratchDatasets import scratchDataset_withname
from models.builder import EncoderDecoder as segmodel
from utils import metrics_scratch
from engine.engine import Engine
from engine.logger import get_logger
from tensorboardX import SummaryWriter


os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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
    def __init__(self):
        super(BCE_Dice, self).__init__()
        self.bce_ration = 0.7
        self.dice_ration = 1-self.bce_ration
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()

    def forward(self, y_pred, y_true):
        bce = self.bce_fn(y_pred, y_true)
        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        return self.bce_ration * bce + self.dice_ration * dice

def test(config,modelpath,threshold=0.5):
    logger = get_logger()
    parser = argparse.ArgumentParser()
    with (Engine(custom_parser=parser) as engine):
        args = parser.parse_args()

        print("=======================================")
        # print(config.tb_dir)
        print("=======================================")

        cudnn.benchmark = True
        seed = config.seed
        if engine.distributed:
            seed = engine.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        if config.dataset_name == 'NEUSD':
            root = "../datasets/scratch_crack/NEUSD_scratch"
        elif config.dataset_name == 'Chip':
            root = "../datasets/scratch_crack/Scratch_Chip"
        elif config.dataset_name == 'steel':
            root = "../datasets/scratch_crack/severstal-steel-scratch"
        elif config.dataset_name == 'car':
            root = "../datasets/scratch_crack/Car_scratch"


        saveroot="../results/VisualResults"

        if not os.path.exists(saveroot):
            os.mkdir(saveroot)
        savepath=os.path.join(saveroot, config.dataset_name)
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        batchsize = config.batch_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        val_dataset = scratchDataset_withname(root, txt="test.txt")
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=4, drop_last=True)

        criterion = BCE_Dice()
        model = segmodel(cfg=config)


        checkpoint = torch.load(modelpath)
        new_state_dict = {}
        for key, value in checkpoint['model'].items():
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=True)

        model.to(device)
        bar2 = tqdm(enumerate(val_loader), total=len(val_loader))
        bar2.set_description('--- eval --- :')
        model.eval()

        with torch.no_grad():
            loss = 0
            acc = 0
            precision = 0
            recall = 0
            f1 = 0
            iou = 0
            miou = 0
            for idx, (img, label, imgpath) in bar2:
                print('testing...',idx)
                img = img.to(device)
                label = label.to(device)
                pred_output = model(img)
                pred = torch.sigmoid(pred_output)

                # threshold = 0.5
                binary_pred = (pred[0] >= threshold).float()
                binary_pred_pil = TF.to_pil_image(binary_pred)
                imgname = imgpath[0].split('/')[-1]
                pred_savepath = os.path.join(savepath, "pred")
                if not os.path.exists(pred_savepath):
                    os.mkdir(pred_savepath)

                binary_pred_pil.save(os.path.join(pred_savepath, imgname[:-4] + ".png"))

                val_loss = criterion(pred_output.view(-1, 1), label.view(-1, 1)) / batchsize
                loss += val_loss
                pred = torch.sigmoid(pred_output)
                ac, p, r, f, = metrics_scratch.f1_loss(label[0], pred)
                acc += ac
                precision += p
                recall += r
                f1 += f
                i, _ = metrics_scratch.iou_score(pred, label[0])
                iou += i
                mi = metrics_scratch.miou(pred, label[0])
                miou += mi
            # [acc,precision,recall,f1,mIoU]
            l = len(val_loader)
            acc = acc / l * 100
            precision = precision / l * 100
            recall = recall / l * 100
            F1 = (2 * precision * recall) / (precision + recall)
            mean_IoU = miou / l * 100
            valloss = loss.item() / l
            acclist = [acc, precision, recall, F1, mean_IoU, valloss]
            print("acc, precision, recall, F1, mean_IoU, valloss:", acclist)

if __name__=="__main__":
    from configs.config_scratch import config
    for threshold in [50]:
        print("threshold", threshold / 100)
        modelpath = '../trainweights/steel.pth'
        test(config,modelpath=modelpath,threshold=threshold / 100)