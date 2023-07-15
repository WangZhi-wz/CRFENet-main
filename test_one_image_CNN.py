import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import cv2
# from loguru import logger
from skimage import io
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
from glob import glob
import torch
import torch.nn.functional as F
from mmseg import __version__
from mmseg.models.segmentors import ColonFormer as UNet
from utils import AvgMeter


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
            image = cv2.resize(image, (224, 224))
            mask = cv2.resize(mask, (224, 224))

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)

epsilon = 1e-7




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
    return 2*((precision*recall)/(precision+recall+epsilon))

def iou_np(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true)+np.sum(y_pred)-intersection
    return intersection/(union+epsilon)

def get_scores_singal(gt, pr):

    mean_precision = precision_np(gt, pr)
    mean_recall = recall_np(gt, pr)
    mean_iou = iou_np(gt, pr)
    mean_dice = dice_np(gt, pr)

    return (mean_iou, mean_dice, mean_precision, mean_recall)


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

def inference(model):
    model.eval()

    loss_record_test = AvgMeter()

    X_test = glob('{}/images/*'.format(datasetpath))
    X_test.sort()
    y_test = glob('{}/masks/*'.format(datasetpath))
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

        res4, res2, res3, res = model(image)
        # res, res2, res3 = model(image)
        # res = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)

        # out image res gt
        img_out = torch.squeeze(image[0]).cpu().numpy()  # 3,320,320
        img_out = np.transpose(img_out, [1, 2, 0])  # 320,320,3

        gt_out = torch.squeeze(gt1[0]).cpu().numpy()  # 320,320
        gt_out[gt_out > 0.5] = 255
        gt_out[gt_out <= 0.5] = 0
        # 添加一个维度
        gt_out = np.expand_dims(gt_out, axis=2)
        # 一维转三维
        gt_out = np.squeeze(gt_out)
        gt_out = [gt_out, gt_out, gt_out]
        gt_out = np.transpose(gt_out, (1, 2, 0))

        resout = torch.squeeze(res[0]).cpu().detach().numpy()
        resout[resout > 0.5] = 255
        resout[resout <= 0.5] = 0
        # 添加一个维度
        predict = np.expand_dims(resout, axis=2)
        # 一维转三维
        predict = np.squeeze(predict)
        predict = [predict, predict, predict]
        predict = np.transpose(predict, (1, 2, 0))

        sep_line = np.ones((224, 10, 3)) * 255
        all_images = [
            img_out * 255,
            sep_line, gt_out,
            sep_line, predict,
        ]

        io.imsave(os.path.join(folder_path, str(i) + ".jpg"), predict)
        io.imsave(os.path.join(folder_path1, str(i) + ".jpg"), gt_out)
        io.imsave(os.path.join(folder_path2, str(i) + ".jpg"), np.concatenate(all_images, axis=1))

        loss = structure_loss(res, gt1)
        loss_record_test.update(loss.data, 1)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pr = res.round()
        iou, dice, precision, recall = get_scores_singal(gt, pr)
        print("\t Number {:0.4f} : dice={:0.4f}  miou={:0.4f}  precision={:0.4f}  recall={:0.4f}"
              .format(i, dice, iou, precision, recall))


        gts.append(gt)
        prs.append(pr)
    mean_iou, mean_dice, mean_precision, mean_recall = get_scores(gts, prs)

    # out
    print("\n" + str(
        epoch) + "\tth model Test Result: loss: {:0.4f}  dice={:0.4f}  miou={:0.4f}  precision={:0.4f}  recall={:0.4f}"
          .format(loss_record_test.show(), mean_dice, mean_iou, mean_precision, mean_recall))


if __name__ == '__main__':

    # modify allocation
    datasetname = 'ISIC811'
    datasetpath = f'./ISIC811/test' # Kvasir CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB CVC-300
    epoch = '3'
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
    backbone = 'b3' # b2 b3
    modelname = 'ConlonFormerB3' # ConlonFormerB2 ConlonFormerB3


    folder_path = f"snapshots/{modelname}/{datasetname}/only_predict"
    folder_path1 = f"snapshots/{modelname}/{datasetname}/only_mask"
    folder_path2 = f"snapshots/{modelname}/{datasetname}/result_all"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(folder_path1):
        os.makedirs(folder_path1)
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)


    # ACSNet
    # from mmseg.models.segmentors import ACSNet as UNet
    # from mmseg.models.segmentors import CRANet as UNet
    # from mmseg.models.segmentors import ResUnetPlusPlus as UNet
    # from mmseg.models.segmentors import UCTransNet as UNet
    # from mmseg.models.segmentors import UNet as UNet
    # from mmseg.models.segmentors import UNetpp as UNet
    from mmseg.models.segmentors.UACANet import UACANet as UNet

    model = UNet(
        # num_classes=1,
    ).cuda()

    weight = "222"
    if weight != '':
        checkpoint = torch.load(f"../../hy-tmp/snapshots/{modelname}/model_save/{epoch}.pth")
        model.load_state_dict(checkpoint['state_dict'], False)

    inference(model)


