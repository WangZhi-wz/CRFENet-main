import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
# from loguru import logger

# from utils.transform import *
from utils import clip_gradient, AvgMeter
from torch.autograd import Variable
import datetime
import torch.nn.functional as F
from torchvision import transforms

# from albumentations.augmentations import transforms
# from albumentations.core.composition import Compose, OneOf

from mmseg import __version__

from thop import profile


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['images']
            mask = augmented['masks']
        else:
            image = cv2.resize(image, (352, 352))
            mask = cv2.resize(mask, (352, 352))

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)


epsilon = 1e-7

def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall

def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision

def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))

def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall*precision/(recall+precision-recall*precision + epsilon)


class FocalLossV1(nn.Module):
    def __init__(self,
                alpha=0.25,
                gamma=2,
                reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wfocal = FocalLossV1()(pred, mask)
    wfocal = (wfocal*weit).sum(dim=(2,3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wfocal + wiou).mean()


#valid----------------------------
def recall_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall


def precision_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision


def dice_np(y_true, y_pred):
    precision = precision_np(y_true, y_pred)
    recall = recall_np(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + epsilon))


def iou_np(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + epsilon)


def get_scores(gts, prs):
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    for gt, pr in zip(gts, prs):
        mean_precision += precision_np(gt, pr)
        mean_recall += recall_np(gt, pr)
        mean_iou += iou_np(gt, pr)
        mean_dice += dice_np(gt, pr)

    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_iou /= len(gts)
    mean_dice /= len(gts)

    return (mean_iou, mean_dice, mean_precision, mean_recall)

def inference(model, args):
    model.eval()

    loss_record_test = AvgMeter()

    X_test = glob('{}/images/*'.format(args.test_path))
    X_test.sort()
    y_test = glob('{}/masks/*'.format(args.test_path))
    y_test.sort()

    test_dataset = Dataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    gts = []
    prs = []
    for i, pack in enumerate(test_loader, start=1):
        image, gt = pack
        gt1 = Variable(gt).cuda()
        gt = gt[0][0]
        gt = np.asarray(gt, np.float32)
        image = image.cuda()

        res = model(image)
        # res, res2, res3, res4 = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)

        loss = structure_loss(res, gt1)
        loss_record_test.update(loss.data, args.batchsize)


        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pr = res.round()
        gts.append(gt)
        prs.append(pr)
    mean_iou, mean_dice, mean_precision, mean_recall = get_scores(gts, prs)

    # out
    print("Test Result: loss: {:0.4f}, dice={}, miou={}, precision={}, recall={} \n"
          .format(loss_record_test.show(), mean_dice, mean_iou, mean_precision, mean_recall))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int,
                        default=30, help='epoch number')
    parser.add_argument('--backbone', type=str,
                        default='b3', help='backbone version')
    parser.add_argument('--init_lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--init_trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--train_path', type=str,
                        default='./datakvasir82/TrainDataset', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='./datakvasir82/TestDataset', help='path to dataset')
    parser.add_argument('--train_save', type=str,
                        default='ConlonFormerB3')
    # ./Kvasir91/train      ./CVC_ClinicDB91/train      ./dataall/TrainDataset      ./datakvasir82/TrainDataset
    # ./Kvasir91/test       ./CVC_ClinicDB91/test       ./dataall/TestDataset       ./datakvasir82/TestDataset

    parser.add_argument('--resume_path', type=str, help='path to checkpoint for resume training',
                        default='')
    parser.add_argument('--weight', type=str,
                        default='snapshots/ConlonFormerB3/last.pth')
    args = parser.parse_args()


    # original
    # from mmseg.models.segmentors import ColonFormer as UNet
    # ssformer Decoder
    # from mmseg.models.segmentors import ColonFormer_ssformer as UNet
    # ssformer mod
    # from mmseg.models.segmentors import ColonFormer_ssformer_mod as UNet
    # U-Net Decoder
    # from mmseg.models.segmentors import ColonFormer_mod as UNet
    # baseline unet
    # from mmseg.models.segmentors import colonformer_ssformer_baseline0301 as UNet
    # mod decoder 2023.03.18
    from mmseg.models.segmentors import colonformer_ssformer_moddecoder as UNet

    model = UNet(backbone=dict(
                    type='mit_{}'.format(args.backbone),
                    style='pytorch'),
                decode_head=dict(
                    type='UPerHead',
                    in_channels=[64, 128, 320, 512],
                    in_index=[0, 1, 2, 3],
                    channels=128,
                    dropout_ratio=0.1,
                    num_classes=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    align_corners=False,
                    decoder_params=dict(embed_dim=768),
                    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
                neck=None,
                auxiliary_head=None,
                train_cfg=dict(),
                test_cfg=dict(mode='whole'),
                pretrained='pretrained/mit_{}.pth'.format(args.backbone)).cuda()


    if args.weight != '':
        checkpoint = torch.load("C:\\Users\\15059\\Desktop\\1.pth")
        model.load_state_dict(checkpoint['state_dict'], False)
    inference(model, args)