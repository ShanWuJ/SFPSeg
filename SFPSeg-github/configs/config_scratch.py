import os
import os.path as osp
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 3407

C.dataset_name = 'steel'


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

C.num_classes = 1
C.class_names =  ['Background', 'Scratch']

# """Image Config"""
C.image_height = 512
C.image_width = 512

C.backbone = 'T' # Remember change the path below.

C.pretrained_model= '../pre_weights/DAMamba-T.pth'   #your pre-trained backbone weights
C.bce_ration = 0.7
C.batch_size = 2

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()