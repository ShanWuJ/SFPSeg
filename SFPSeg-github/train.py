import os
import sys
import time
import argparse
import os.path as osp
from types import SimpleNamespace
import pandas
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from dataloader.scratchDatasets import scratchDataset
from models.builder import EncoderDecoder as segmodel, BCE_Dice
from utils import metrics_scratch
from utils.lr_policy import  MultiGroupWarmUpPolyLR
from engine.engine import Engine
from utils.pyt_utils import all_reduce_tensor
from engine.logger import get_logger
from tensorboardX import SummaryWriter
from ptflops import get_model_complexity_info


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def parse_args():
    """Parse parameters"""
    parser = argparse.ArgumentParser(description='model train set')

    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    # dataset config
    parser.add_argument('--dataset-name', type=str, default='car',
                        choices=['NEUSD', 'Chip', 'steel', 'car'], help='dataset name')
    parser.add_argument('--num-train-imgs', type=int, default=None, help='train images number')
    parser.add_argument('--num-eval-imgs', type=int, default=None,  help='eval images number')

    # model config
    parser.add_argument('--num-classes', type=int, default=1, help='classes')
    parser.add_argument('--class-names', type=str, nargs='+', default=['Background', 'Scratch'], help='classes name')
    parser.add_argument('--image-height', type=int, default=512, help='iamge height')
    parser.add_argument('--image-width', type=int, default=512, help='iamge width')
    parser.add_argument('--backbone', type=str, default='T', help='backbone')
    parser.add_argument('--pretrained-model', type=str, default='../pre_weights/DAMamba-T.pth',
                        help='pre-trained model path')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adamw', 'SGD'], help='optimizer')

    # train config
    # parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--lr-power', type=float, default=0.9, help='lr decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--real-nepochs', type=int, default=150, help='epoches')
    parser.add_argument('--warm-up-epoch', type=int, default=4, help='warm up epoches')
    parser.add_argument('--bce-ration', type=int, default=0.7, help='warm up epoches')

    # checkpoint and log
    parser.add_argument('--checkpoint-start-epoch', type=int, default=20, help='epoch of save checkpoint')
    parser.add_argument('--checkpoint-step', type=int, default=1, help='step of save checkpoint')
    parser.add_argument('--nepochs', type=int, default=185, help='pre-set epoches')
    parser.add_argument('--log-dir', type=str, default=None, help='log')

    # Parse the parameters
    args = parser.parse_args()
    return args

def create_config(args):
    """Create a configuration object based on the parameters"""
    # use SimpleNamespace
    C = SimpleNamespace()

    # 复制所有参数
    for key, value in vars(args).items():
        setattr(C, key, value)

    # set image number of different datasets
    if C.dataset_name == 'NEUSD':
        C.num_train_imgs = 800
        C.num_eval_imgs = 100
    elif C.dataset_name == 'Chip':
        C.num_train_imgs = 2192
        C.num_eval_imgs = 549
    elif C.dataset_name == 'steel':
        C.num_train_imgs = 4120
        C.num_eval_imgs = 1030
    elif C.dataset_name == 'car':
        C.num_train_imgs = 268
        C.num_eval_imgs = 67

    # count itermsof each epoch
    C.niters_per_epoch = C.num_train_imgs // C.batch_size

    # set log_dir
    if C.log_dir is None:
        if C.dataset_name == 'NEUSD':
            C.log_dir = osp.abspath('log_final/NEUSD_Aug/' + 'log_' + C.dataset_name + '_' + C.backbone + '_0')
        elif C.dataset_name == 'Chip':
            C.log_dir = osp.abspath('log_final/ChipB/' + 'log_' + C.dataset_name + '_' + C.backbone + '_0')
        elif C.dataset_name == 'steel':
            C.log_dir = osp.abspath('log_final/steel/' + 'log_' + C.dataset_name + '_' + C.backbone + '_0')
        elif C.dataset_name == 'car':
            C.log_dir = osp.abspath('log_final/car/' + 'log_' + C.dataset_name + '_' + C.backbone + '_0')

    original_log_dir = C.log_dir
    counter = 0
    while os.path.exists(C.log_dir):
        if C.log_dir[-1].isdigit():
            base = C.log_dir[:-1]
            counter = int(C.log_dir[-1]) + 1
            C.log_dir = base + str(counter)
        else:
            C.log_dir = original_log_dir + '_' + str(counter)
            counter += 1

    C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
    C.log_dir_link = C.log_dir
    C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))
    return C


def main(config):
    logger = get_logger()
    parser = argparse.ArgumentParser()
    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        print("=======================================")
        print(config.tb_dir)
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
            root = "../datasets/scratch_crack/Scratch_ChipB"
        elif config.dataset_name == 'steel':
            root = "../datasets/scratch_crack/severstal-steel-scratch"
        elif config.dataset_name == 'car':
            root = "../datasets/scratch_crack/Car_scratch"

        batchsize = config.batch_size
        trainset = scratchDataset(root, txt="train.txt")
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                                   shuffle=True, num_workers=4, drop_last=True)

        val_dataset = scratchDataset(root, txt="test.txt")
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=4, drop_last=True)

        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
            generate_tb_dir = config.tb_dir + '/tb'
            tb = SummaryWriter(log_dir=tb_dir)
            engine.link_tb(tb_dir, generate_tb_dir)
        criterion = BCE_Dice(config)

        # if engine.distributed:
        #     GroupNorm = nn.SyncBatchNorm
        # else:
        norm = nn.GroupNorm

        model = segmodel(cfg=config)

        # 1. 定义参数组（不同部分使用不同学习率）
        param_groups = [
            {'params': model.encoder.parameters(), 'lr': 0.0001, 'name': 'encoder'},
            {'params': model.decoder.parameters(), 'lr': 0.001, 'name': 'decoder'},
            {'params': model.bff.parameters(), 'lr': 0.001, 'name': 'bff'},
            {'params': model.seghead.parameters(), 'lr': 0.001, 'name': 'seghead'},
            {'params': model.up.parameters(), 'lr': 0.001, 'name': 'up'},
        ]

        # 2. 创建优化器
        if config.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=config.weight_decay)
        elif config.optimizer == 'SGD':
            optimizer = torch.optim.SGD(param_groups, momentum=config.momentum,
                                        weight_decay=config.weight_decay)
        else:
            raise NotImplementedError

        # config lr policy
        # 3. 计算总迭代次数
        total_iters = config.nepochs * config.niters_per_epoch
        warmup_steps = config.warm_up_epoch * config.niters_per_epoch

        # 4. 获取每个参数组的初始学习率
        base_lrs = [group['lr'] for group in param_groups]

        # 5. 创建多参数组调度器
        scheduler = MultiGroupWarmUpPolyLR(
            optimizer=optimizer,
            base_lrs=base_lrs,
            lr_power=config.lr_power,
            total_iters=total_iters,
            warmup_steps=warmup_steps
        )

        if engine.distributed:
            logger.info('.............distributed training.............')
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                # model.cuda()
                model.to(device)
                model = DistributedDataParallel(model, device_ids=[0],
                                                output_device=0, find_unused_parameters=False)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)

        model_name = 'SFPSeg'
        flops, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=True)
        print("%s |%s |%s" % (model_name, flops, params))

        engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
        if engine.continue_state_object:
            engine.restore_checkpoint()


        optimizer.zero_grad()
        model.train()
        logger.info('begin trainning:')

        best_mean_iou = 0.0  # Track the best mean IoU for model saving
        best_epoch = 100000  # Track the epoch with the best mean IoU for model saving

        describepath = os.path.join(config.log_dir, "describe.txt")
        with open(describepath, "w") as f:
            f.write("dataset=" + config.dataset_name +"\n" + "encoder=" + config.backbone + "\n" +
                    "lr_encoder=" + str(optimizer.param_groups[0]['lr']) + "\n" +   "lr_other=" + str(optimizer.param_groups[1]['lr'])+ "\n" +
                    "epoches=" + str(config.real_nepochs) + "\n" + "BCE_ration=" + str(config.bce_ration) + "\n")
            f.close()

        global_iter = 0
        for epoch in range(engine.state.epoch, config.real_nepochs + 1):
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                        bar_format=bar_format)
            dataloader = iter(train_loader)
            sum_loss = 0
            for idx in pbar:
                # 更新学习率
                scheduler.step(global_iter)

                engine.update_iteration(epoch, idx)
                (imgs, gts) = next(dataloader)
                imgs = imgs.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)
                loss = model(imgs, gts)
                if engine.distributed:
                    reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_iter += 1

                lr_backbone = optimizer.param_groups[0]['lr']
                lr_other = optimizer.param_groups[1]['lr']


                if engine.distributed:
                    if dist.get_rank() == 0:
                        sum_loss += reduce_loss.item()
                        print_str = 'Epoch {}/{}'.format(epoch, config.real_nepochs) \
                                    + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                                    + ' lr_encoder=%.4e' % lr_backbone + ' lr_other=%.4e' % lr_other \
                                    + ' total_loss=%.4f' % (sum_loss / (idx + 1))
                        pbar.set_description(print_str, refresh=False)
                else:
                    sum_loss += loss
                    print_str = 'Epoch {}/{}'.format(epoch, config.real_nepochs) \
                                + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                                + ' lr_encoder=%.4e' % lr_backbone + ' lr_other=%.4e' % lr_other \
                                + 'total_loss=%.4f' % (sum_loss / (idx + 1))
                    pbar.set_description(print_str, refresh=False)
                del loss

            if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
                tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

            if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (
                    epoch == config.nepochs):
                if engine.distributed and (engine.local_rank == 0):
                    engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)
                elif not engine.distributed:
                    engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)

            torch.cuda.empty_cache()
            if (epoch >= config.checkpoint_start_epoch) and (
                    epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                bar2 = tqdm(enumerate(val_loader), total=len(val_loader))
                bar2.set_description('Epoch %d --- eval --- :' % epoch)
                model.eval()
                with torch.no_grad():
                    loss = 0
                    acc = 0
                    precision = 0
                    recall = 0
                    f1 = 0
                    iou = 0
                    miou = 0
                    for idx, (img, label) in bar2:
                        img = img.to(device)
                        label = label.to(device)
                        pred_output = model(img)
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
                    # mf1 = f1 / l * 100
                    F1 = (2 * precision * recall) / (precision + recall)
                    # PIoU = iou / l * 100
                    mean_IoU = miou / l * 100
                    valloss = loss.item() / l
                    acclist = [acc, precision, recall, F1, mean_IoU, valloss, epoch]
                    print("valloss", valloss)
                    data_valloss = pandas.DataFrame([valloss])
                    data_valloss.to_csv(config.log_dir + '/valloss.csv', mode='a', header=False, index=False)
                    print("acc", acclist)
                    data_acc = pandas.DataFrame([acclist])
                    data_acc.to_csv(config.log_dir + '/acc.csv', mode='a', header=False, index=False)
                    all_acc=pandas.read_csv(config.log_dir + '/acc.csv',header=None)
                    sort_all_acc=all_acc.sort_values(by=all_acc.columns[-3],ascending=False)
                    sort_all_acc.to_csv(config.log_dir + '/acc_sort.csv',header=False, index=False)
                    print('mean_IoU:', mean_IoU)

                    # Determine if the model performance improved
                    if mean_IoU > best_mean_iou:
                        # If the model improves, remove the saved checkpoint for this epoch
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                        best_epoch = epoch
                        best_mean_iou = mean_IoU
                    else:
                        # If the model does not improve, remove the saved checkpoint for this epoch
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                model.train()

if __name__=="__main__":
    args = parse_args()
    config = create_config(args)
    main(config)

