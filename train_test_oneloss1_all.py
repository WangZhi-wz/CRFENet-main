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

    # bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    wdice = 1 - (2 * inter + 1)/(union+1)
    # wbce = (weit * bce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # return (wfocal + wdice + wiou).mean()
    # return (wbce + wdice).mean()
    # return (wbce + wiou).mean()
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

    X_test = glob('{}/images/*'.format(args.test_data_path))
    X_test.sort()
    y_test = glob('{}/masks/*'.format(args.test_data_path))
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
        # print(loss)
        # print(loss.data)
        # print(loss_record_test.show())


        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pr = res.round()
        gts.append(gt)
        prs.append(pr)
    mean_iou, mean_dice, mean_precision, mean_recall = get_scores(gts, prs)

    # tensorboard
    tags1 = ["test_loss", "test_Dice", "test_IoU", "test_precision", "test_recall"]
    tb_writer.add_scalar(tags1[0], loss_record_test.show(), epoch)
    tb_writer.add_scalar(tags1[1], mean_dice, epoch)
    tb_writer.add_scalar(tags1[2], mean_iou, epoch)
    tb_writer.add_scalar(tags1[3], mean_precision, epoch)
    tb_writer.add_scalar(tags1[4], mean_recall, epoch)

    # out
    print("Test Result: loss: {:0.4f}, dice={}, miou={}, precision={}, recall={} \n"
                .format(loss_record_test.show(), mean_dice, mean_iou, mean_precision, mean_recall))
    f = f"{save_path}result/train.txt"
    with open(f, "a") as filenametest:  # ”w"代表着每次运行都覆盖内容
        filenametest.write("\nTest Result: loss: {:0.4f}, dice={}, miou={}, precision={}, recall={} \n"
                .format(loss_record_test.show(), mean_dice, mean_iou, mean_precision, mean_recall))
    filenametest.close()


# train
def train(train_loader, model, optimizer, epoch, lr_scheduler, args):

    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    dice, iou = AvgMeter(), AvgMeter()
    # debug 0.pth
    with torch.autograd.set_detect_anomaly(True):
        for i, pack in enumerate(train_loader, start=1):
            if epoch <= 1:
                    optimizer.param_groups[0]["lr"] = (epoch * i) / (1.0 * total_step) * args.init_lr
                    print("learning rate:{:0.7f}".format(optimizer.param_groups[0]["lr"]))
            else:
                lr_scheduler.step()

            for rate in size_rates: 
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(args.init_trainsize*rate/32)*32)
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                map4 = model(images)
                loss = structure_loss(map4, gts)
                # map1 = F.upsample(map1, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # map2 = F.upsample(map2, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # map3 = F.upsample(map3, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # map4 = F.upsample(map4, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # map4, map3, map2, map1 = model(images)
                # loss = structure_loss(map1, gts) + structure_loss(map2, gts) + structure_loss(map3, gts) + structure_loss(map4, gts)
                # with torch.autograd.set_detect_anomaly(True):
                #loss = nn.functional.binary_cross_entropy(map1, gts)
                # ---- metrics ----
                dice_score = dice_m(map4, gts)
                iou_score = iou_m(map4, gts)
                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, args.clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record.update(loss.data, args.batchsize)
                    dice.update(dice_score.data, args.batchsize)
                    iou.update(iou_score.data, args.batchsize)

            # ---- train visualization ----
            if i == total_step:
                # tensorboard
                tags = ["train_loss"]
                tb_writer.add_scalar(tags[0], loss_record.show(), epoch)
                # out
                print('\nTraining Epoch [{:03d}/{:03d}]\nloss: {:0.4f}, dice: {:0.4f}, iou: {:0.4f}'.
                        format(epoch, args.num_epochs, loss_record.show(), dice.show(), iou.show()))
                f = f"{save_path}/result/train.txt"
                with open(f, "a") as filenametest:  # ”w"代表着每次运行都覆盖内容
                    filenametest.write('\nTraining Epoch [{:03d}/{:03d}]\nloss: {:0.4f}, dice: {:0.4f}, iou: {:0.4f}'.
                        format(epoch, args.num_epochs, loss_record.show(), dice.show(), iou.show()))
                filenametest.close()

    ckpt_path = save_model_path + str(epoch) + '.pth'
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, ckpt_path)

    if args.weight != '':
        checkpoint = torch.load(save_model_path + str(epoch) + ".pth")
        model.load_state_dict(checkpoint['state_dict'])
    inference(model, args)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    setup_seed(3407)

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

    parser.add_argument('--train_data_path', type=str,
                        default='./dataall/TrainDataset', help='path to dataset')
    parser.add_argument('--test_data_path', type=str,
                        default='./dataall/TestDataset/CVC-ClinicDB', help='path to dataset')

    # kvasir82  ETIS-LaribPolypDB82  CVC_ClinicDB811  CVC-ColonDB811  ISIC2017  ISIC811  bowl  GLas  COVID
    #  kvasir82                 ./kvasir82/train   ./kvasir82/test
    #  ETIS-LaribPolypDB82      ./ETIS-LaribPolypDB82/train   ./ETIS-LaribPolypDB82/test
    #  CVC_ClinicDB811          ./CVC_ClinicDB811/train    ./CVC_ClinicDB811/test
    #  CVC-ColonDB811           ./CVC-ColonDB811/train   ./CVC-ColonDB811/test
    #  ISIC2017                 ./ISIC2017/train   ./ISIC2017/test
    #  ISIC811                  ./ISIC811/train   ./ISIC811/test
    #  bowl                     ./bowl/train   ./bowl/test
    #  GLas                     ./GLas/train   ./GLas/test
    #  COVID                    ./COVID/train   ./COVID/test
    # ./dataall/TrainDataset   ./dataall/TestDataset
    parser.add_argument('--train_save', type=str,
                        default='ConlonFormerB3')
    parser.add_argument('--resume_path', type=str, help='path to checkpoint for resume training',
                        default='')
    parser.add_argument('--weight', type=str,
                        default='snapshots/ConlonFormerB2/last.pth')
    args = parser.parse_args()

    save_path = '../../hy-tmp/snapshots/{}/'.format(args.train_save)
    save_model_path = '../../hy-tmp/snapshots/{}/model_save/'.format(args.train_save)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path, exist_ok=True)
    else:
        print("Save path existed")

    train_img_paths = []
    train_mask_paths = []
    train_img_paths = glob('{}/images/*'.format(args.train_data_path))
    train_mask_paths = glob('{}/masks/*'.format(args.train_data_path))
    train_img_paths.sort()
    train_mask_paths.sort()

    train_dataset = Dataset(train_img_paths, train_mask_paths)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )




    total_step = len(train_loader)

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
    # from mmseg.models.segmentors import colonformer_ssformer_moddecoder as UNet
    # mod decoder CA 2023.03.22
    # from mmseg.models.segmentors import colonformer_ssformer_moddecoder_CA as UNet
    # mod decoder AFF 2023.03.28
    # from mmseg.models.segmentors import colonformer_ssformer_moddecoder_AFF as UNet
    # mod decoder colonformer_ssformer_baseline0301_AFF
    # from mmseg.models.segmentors import colonformer_ssformer_baseline0301_AFF as UNet
    # mod decoder colonformer_ssformer_0403_AFF_BG
    # from mmseg.models.segmentors import colonformer_ssformer_0403_AFF_BG as UNet
    # mod decoder colonformer_ssformer_0403_AFF_FM
    # from mmseg.models.segmentors import colonformer_ssformer_0403_AFF_FM as UNet
    # mode decoder colonformer_ssformer_baseline0301_AFF_LCA
    # from mmseg.models.segmentors import colonformer_ssformer_baseline0301_AFF_LCA as UNet
    # mod decoder colonformer_ssformer_0403_AFF_BG + fuallattention
    # from mmseg.models.segmentors import colonformer_ssformer_0403_AFF_BG_FA as UNet
    # mod decoder colonformer_ssformer_0403_AFF_BG + res2net
    # from mmseg.models.segmentors import colonformer_ssformer_0403_AFF_BGRes2Net as UNet
    # mod decoder colonformer_ssformer_baseline0301_AFF_DP
    # from mmseg.models.segmentors import colonformer_ssformer_baseline0301_AFF_DP as UNet
    # from mmseg.models.segmentors import colonformer_ssformer_baseline0301_AFF_DPmod as UNet
    # from mmseg.models.segmentors import colonformer_ssformer_baseline0301_AFF_DP0505 as UNet
    # from mmseg.models.segmentors import colonformer_ssformer_baseline0301_AFF_DP0505_se as UNet
    # from mmseg.models.segmentors import colonformer_ssformer_baseline0301_AFF_DP0505_onlyencoder as UNet
    # from mmseg.models.segmentors import colonformer_mitbaseline as UNet
    # from mmseg.models.segmentors import medt_net as UNet
    from mmseg.models.segmentors.colonformer_segformer import ColonFormer_segformer as UNet

    model = UNet(backbone=dict(
                    # mit
                    type='mit_{}'.format(args.backbone),  # pvt_v2_b2
                    # pvt_v2
                    # type='pvt_v2_{}'.format(args.backbone),
                    # ResNet
                    # type='ResNet',
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
                # ResNet
                # pretrained='pretrained/resnet34-333f7ec4.pth'
                pretrained='pretrained/mit_{}.pth'.format(args.backbone)).cuda()


    # ---- optimizer and lr_scheduler ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                        T_max=len(train_loader)*args.num_epochs,
                                        eta_min=args.init_lr/1000)

    start_epoch = 1
    if args.resume_path != '':
        checkpoint = torch.load(args.resume_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # ---- flops and params ----
    input = torch.randn(1, 3, 352, 352).to('cuda')
    macs, params = profile(model, inputs=(input,))


    # 将模型名写入模型结果文件
    # time
    x = datetime.datetime.now()

    # tensorboard
    tb_writer = SummaryWriter(
        log_dir=f"{save_path}/result/logs_" + str(x.strftime('%Y_%m_%d_%H_%M_%S')))
    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 352, 352)).cuda()
    tb_writer.add_graph(model, init_img)

    # out
    print(f"\n\n\n\n\nstart training: {args.train_save}  macs: {macs / 1000000000}, params: {params / 1000000}, time:{x.strftime('%Y-%m-%d %H:%M:%S')}")
    f = f"{save_path}/result/train.txt"
    with open(f, "a") as filenametest:  # ”w"代表着每次运行都覆盖内容
        filenametest.write(f"\n\n\n\n\nstart training: {args.train_save}  macs: {macs / 1000000000}, params: {params / 1000000}, time:{x.strftime('%Y-%m-%d %H:%M:%S')}")
    filenametest.close()


    for epoch in range(start_epoch, args.num_epochs+1):
        train(train_loader, model, optimizer, epoch, lr_scheduler, args)
