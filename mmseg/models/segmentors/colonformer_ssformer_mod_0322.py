import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import numpy as np
import cv2

from .lib.conv_layer import Conv, BNPReLU
from .lib.axial_atten import AA_kernel
from .lib.context_module import CFPModule

# CFM weight_init
def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


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


# CFM Module
class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        channel = 64
        self.conv1h = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(channel)
        self.conv2h = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(channel)
        self.conv3h = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(channel)
        self.conv4h = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(channel)

        self.conv1v = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(channel)
        self.conv2v = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(channel)
        self.conv3v = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(channel)
        self.conv4v = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn4v   = nn.BatchNorm2d(channel)

    def forward(self, left, down):  # left 1 64 20 20  down 1 64 10 10
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left )), inplace=True) # 1 64 20 20
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True) # 1 64 20 20
        out1v = F.relu(self.bn1v(self.conv1v(down )), inplace=True) # 1 64 20 20
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True) # 1 64 20 20
        fuse  = out2h*out2v
        out3h = F.relu(self.bn3h(self.conv3h(fuse )), inplace=True)+out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse )), inplace=True)+out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        return out4h, out4v

    def initialize(self):
        weight_init(self)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.dropout = nn.Dropout2d(0.5)
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x


class Decoder(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, dims, dim, class_num=2):
        super(Decoder, self).__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = 128

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse3 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        self.linear_pred = Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

        self.cfm3 = CFM()
        self.cfm2 = CFM()
        self.cfm1 = CFM()

        self.outconv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(embedding_dim, self.num_classes, 1))

        self.outconvfinal = nn.Sequential(

            nn.Dropout(0.1),
            nn.Conv2d(embedding_dim, self.num_classes, 1))

        self.conv = Conv2d(embedding_dim, embedding_dim, kernel_size=1)

        self.sideout1 = SideoutBlock(embedding_dim, 1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        # c1 1x64x88x88  c2 1x128x44x44  c3 1x256x22x22  c4 1x512x11x11
        n, _, h, w = c4.shape

        # LA Module-----------------------------------------------------------------------------------------------------
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])  # 1x64x88x88 -> 1x256x88x88

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])  # 1x128x44x44 -> 1x256x44x44
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)  # 1x256x88x88

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])  # 1x320x22x22 -> 1x256x22x22
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)  # 1x256x88x88

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])  # 1x512x11x11 -> 1x256x11x11
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)  # 1x256x88x88


        # Linear fuse 1 stage-------------------------------------------------------------------------------------------
        L12 = torch.cat([_c2, _c1], dim=1)
        L12 = self.linear_fuse1(L12)  # 1x256x88x88

        L23 = torch.cat([_c3, _c2], dim=1)
        L23 = self.linear_fuse2(L23)  # 1x256x88x88

        L34 = torch.cat([_c4, _c3], dim=1)
        L34 = self.linear_fuse3(L34)  # 1x256x88x88

        # CFM 1 stage---------------------------------------------------------------------------------------------------
        # cfm3: L34 and _c4
        # L34, out3 = self.cfm3(L34, _c4)
        # cfm2: L23 and out3
        # L23, out2 = self.cfm2(L23, out3)
        # cfm1: L12 and out2
        # L12, pred1 = self.cfm1(L12, out2)
        L34 = self.conv(L34 + _c4)
        L23 = self.conv(L23 + L34)
        L12 = self.conv(L12 + L23)

        # sideout
        out1 = self.sideout1(L12)
        # LCA
        score = torch.sigmoid(out1)
        dist = torch.abs(score - 0.5)
        att1 = 1 - (dist / 0.5)


        # Linear fuse 2 stage-------------------------------------------------------------------------------------------
        L2_12 = torch.cat([L23, L12], dim=1)
        L2_12 = self.linear_fuse1(L2_12)  # 1x256x88x88

        L2_23 = torch.cat([L34, L23], dim=1)
        L2_23 = self.linear_fuse2(L2_23)  # 1x256x88x88

        # feedback
        L2_12 = L2_12 * att1 + L2_12
        L2_23 = L2_23 * att1 + L2_12

        # CFM 2 stage---------------------------------------------------------------------------------------------------
        # cfm22: L2_23 and _c4
        # L2_23, out2 = self.cfm2(L2_23, _c4)
        # cfm21: L2_12 and out2
        # L2_12, pred2 = self.cfm1(L2_12, out2)
        L2_23 = self.conv(L2_23 + _c4)
        L2_12 = self.conv(L2_12 + L2_23)

        # sidout
        out2 = self.sideout1(L2_12)
        # LCA
        score = torch.sigmoid(out2)
        dist = torch.abs(score - 0.5)
        att2 = 1 - (dist / 0.5)

        # Linear fuse 3 stage-------------------------------------------------------------------------------------------
        L3_12 = torch.cat([L2_23, L2_12], dim=1)
        L3_12 = self.linear_fuse1(L3_12)

        # out
        # pred3 = L3_12 + pred2
        L3_12 = L3_12 * att2 + L3_12



        # pred3 = self.outconvfinal(L3_12)
        # pred2 = self.outconv(pred2)
        # pred1 = self.outconv(pred1)

        L3_12 = self.dropout(L3_12)
        pred3 = self.linear_pred(L3_12)

        # return pred3, pred2, pred1
        return pred3, out2, out1


@SEGMENTORS.register_module()
class colonformer_ssformer_mod_0322(nn.Module):
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
        super(colonformer_ssformer_mod_0322, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=256, class_num=1)
        self.num_classes = self.decode_head.num_classes
        self.backbone.init_weights(pretrained=pretrained)

    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  # 1x64x88x88
        x2 = segout[1]  # 1x128x44x44
        x3 = segout[2]  # 1x320x22x22
        x4 = segout[3]  # 1x512x11x11

        # Uper Decoder
        # predict out1
        pred3, pred2, pred1 = self.decode_head.forward([x1, x2, x3, x4]) # 1x1x88x88
        lateral_map_3 = F.interpolate(pred3, scale_factor=4, mode='bilinear') # 1x1x352x352
        lateral_map_2 = F.interpolate(pred2, scale_factor=4, mode='bilinear')  # 1x1x352x352
        lateral_map_1 = F.interpolate(pred1, scale_factor=4, mode='bilinear')  # 1x1x352x352

        # pred3 = self.decode_head.forward([x1, x2, x3, x4]) # 1x1x88x88
        # lateral_map_3 = F.interpolate(pred3, scale_factor=4, mode='bilinear') # 1x1x352x352


        return lateral_map_3, lateral_map_2, lateral_map_1
