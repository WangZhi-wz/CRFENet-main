import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d, init
from mmseg.core import add_prefix
from mmseg.ops import resize
from .odconv import ODConv2d
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import numpy as np
import cv2

from .lib.conv_layer import Conv, BNPReLU
from .lib.axial_atten import AA_kernel
from .lib.context_module import CFPModule
import torchvision.models as models


class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x) # 1x256x11x11 1x256x22x22 1x256x44x44 1x256x88x88
        x = x.flatten(2).transpose(1, 2) # 1x121x256 1x484x256 1x1936x256 1x7744x256
        return x



class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv3(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def odconv3x3(in_planes, out_planes, stride=1, reduction=0.25, kernel_num=4):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)

def odconv5x5(in_planes, out_planes, stride=3, reduction=0.25, kernel_num=4):
    return ODConv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2,
                    reduction=reduction, kernel_num=kernel_num)


def odconv1x1(in_planes, out_planes, stride=1, reduction=0.25, kernel_num=4):
    return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    reduction=reduction, kernel_num=kernel_num)

class ODconv(nn.Module):
    def __init__(self, dim, expansion=4, stride=1, reduction=0.25, kernel_num=4): # 0.0625
        super(ODconv, self).__init__()
        # ODConv
        self.conv1 = odconv1x1(dim, dim//expansion, reduction=reduction, kernel_num=kernel_num)
        self.bn1 = nn.BatchNorm2d(dim//expansion)
        self.conv2 = odconv3x3(dim//expansion, dim//expansion, stride, reduction=reduction, kernel_num=kernel_num)
        self.bn2 = nn.BatchNorm2d(dim//expansion)
        self.conv3 = odconv1x1(dim//expansion, dim, reduction=reduction, kernel_num=kernel_num)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """


    def __init__(self, dims, dim, class_num=2):
        super(Decoder, self).__init__()
        self.num_classes = class_num
        init_feat = 32

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim64, embedding_dim128, embedding_dim320, embedding_dim512 = dim[0], dim[1], dim[2], dim[3]
        # dims = [64, 128, 320, 512], dim = [64, 128, 320, 512]

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim512)
        self.linear_c3 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim320)
        self.linear_c2 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim128)
        self.linear_c1 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim64)
        #
        self.linear_fuse3 = ConvModule(in_channels=embedding_dim320 * 2, out_channels=embedding_dim320, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim128 * 2, out_channels=embedding_dim128, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim64 * 2, out_channels=embedding_dim64, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        #
        self.linear_pred = Conv2d(embedding_dim64, self.num_classes, kernel_size=1)
        self.linear_pred2 = nn.Sequential(Conv2d(embedding_dim128, embedding_dim64, kernel_size=1),
                                          Conv2d(embedding_dim64, self.num_classes, kernel_size=1))
        self.linear_pred3 = nn.Sequential(Conv2d(embedding_dim320, embedding_dim64, kernel_size=1),
                                          Conv2d(embedding_dim64, self.num_classes, kernel_size=1))

        self.upconv3 = up_conv(512, 320)
        self.upconv2 = up_conv(320, 128)
        self.upconv1 = up_conv(128, 64)

        self.convcc3 = Conv2d(640, 320, kernel_size=1)
        self.convcc2 = Conv2d(256, 128, kernel_size=1)
        self.convcc1 = Conv2d(128, 64, kernel_size=1)

        self.conv = conv3(512, 512)

        self.odconv4 = ODconv(512)
        self.odconv3 = ODconv(320)
        self.odconv2 = ODconv(128)
        self.odconv1 = ODconv(64)


        self.dropout = nn.Dropout(0.1)


    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape


        # _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])   # 1x512x11x11 -> 1x512x11x11

        # _c4 = self.conv(c4)
        # c4 = self.odconv4(c4)
        # c3 = self.odconv3(c3)
        # c2 = self.odconv2(c2)
        # c1 = self.odconv1(c1)

        _c3 = self.upconv3(c4) # 1x320x22x22
        _c3 = torch.cat((_c3, c3), dim=1) # 1x640x22x22
        _c3 = self.convcc3(_c3) # 1x320x22x22

        _c2 = self.upconv2(_c3) # 1x128x44x44
        _c2 = torch.cat((_c2, c2), dim=1) # 1X256X44X44
        _c2 = self.convcc2(_c2) # 1x128x44x44

        _c1 = self.upconv1(_c2) # 1X64X88X88
        _c1 = torch.cat((_c1, c1), dim=1) # 1X128X88X88
        _c1 = self.convcc1(_c1) # 1X64X88X88

        x = self.dropout(_c1)
        x = self.linear_pred(x)  # 1x1x88x88

        return x



@SEGMENTORS.register_module()
class colonformer_ssformer_baseline0301(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,  # backbones -> mix_transformer
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(colonformer_ssformer_baseline0301, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=[64, 128, 320, 512], class_num=1)
        # self.decode_head = builder.build_head(decode_head)
        # self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        # self.decode_head.init_weights()


        
    def forward(self, x):
        segout = self.backbone(x)

        x1 = segout[0]  #  1x64x88x88
        x2 = segout[1]  # 1x128x44x44
        x3 = segout[2]  # 1x320x22x22
        x4 = segout[3]  # 1x512x11x11

        # Uper Decoder
        # predict out1
        decoder_1 = self.decode_head.forward([x1, x2, x3, x4])
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear') # 1x1x352x352

        return lateral_map_1