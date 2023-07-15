import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from skimage import io
from torch.nn import Conv2d, UpsamplingBilinear2d, init, Parameter
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
        x = self.proj(x)  # 1x256x11x11 1x256x22x22 1x256x44x44 1x256x88x88
        x = x.flatten(2).transpose(1, 2)  # 1x121x256 1x484x256 1x1936x256 1x7744x256
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


# AFF Module
# ----------------------------------------------------------------------------------------------------------------------
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


# iAFF Module
# ----------------------------------------------------------------------------------------------------------------------
class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
class iAFF_method1(nn.Module):

    def __init__(self, channels=64, r=4):
        super(iAFF_method1, self).__init__()
        inter_channels = int(channels // r)
        # Global
        # 全局注意力
        self.global_attmax = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )
        self.global_attavg = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )

        # Local
        # 本地注意力
        self.local_att1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att3 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        # self.w = nn.Parameter(torch.ones(4))

    def forward(self, x, residual):
        xl1 = self.local_att1(x)
        xl2 = self.local_att3(x)
        xg1 = self.global_attmax(x)
        xg2 = self.global_attavg(x)
        xlg = xl1 + xl2 + xg1 + xg2
        wei = self.sigmoid(xlg)
        xi = x * wei

        xl11 = self.local_att1(residual)
        xl22 = self.local_att3(residual)
        xg11 = self.global_attmax(residual)
        xg22 = self.global_attavg(residual)

        xlg2 = xl11 + xl22 + xg11 + xg22
        wei2 = self.sigmoid(xlg2)
        xu = residual * wei2

        xo = xi + xu

        return xo
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
class iAFF_method2(nn.Module):

    def __init__(self, channels=64, r=4):
        super(iAFF_method2, self).__init__()
        inter_channels = int(channels // r)
        # Global
        # 全局注意力
        self.global_attmax = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )
        self.global_attavg = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )

        # Local
        # 本地注意力
        self.local_att1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att3 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):

        x = x + residual

        xl1 = self.local_att1(x)
        xl2 = self.local_att3(x)
        xg1 = self.global_attmax(x)
        xg2 = self.global_attavg(x)
        xlg = xl1 + xl2 + xg1 + xg2
        wei = self.sigmoid(xlg)
        xi = x * wei

        return xi
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
class iAFF_method3(nn.Module):

    def __init__(self, channels=64, r=4):
        super(iAFF_method3, self).__init__()
        inter_channels = int(channels // r)
        # Global
        # 全局注意力
        self.global_attmax = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )
        self.global_attavg = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )

        # Local
        # 本地注意力
        self.local_att1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att3 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, y, x):

        xl1 = self.local_att1(y)
        xl2 = self.local_att3(y)
        xg1 = self.global_attmax(y)
        xg2 = self.global_attavg(y)
        xlg = xl1 + xl2 + xg1 + xg2
        wei = self.sigmoid(xlg)
        xi = x * wei
        xo = xi + y

        return xo
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
class iAFF_method4(nn.Module):

    def __init__(self, channels=64, r=4):
        super(iAFF_method4, self).__init__()
        inter_channels = int(channels // r)
        # Global
        # 全局注意力
        self.global_attmax = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )
        self.global_attavg = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )

        # Local
        # 本地注意力
        self.local_att1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att3 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, y, x):

        xl1 = self.local_att1(y)
        xl2 = self.local_att3(y)
        xg1 = self.global_attmax(y)
        xg2 = self.global_attavg(y)
        xlg = xl1 + xl2 + xg1 + xg2
        wei = self.sigmoid(xlg)
        xi = y * wei
        xo = xi + x

        return xo
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
class iAFF_method5(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF_method5, self).__init__()
        inter_channels = int(channels // r)
        # Global
        # 全局注意力
        self.global_attmax = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )
        self.global_attavg = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )

        # Local
        # 本地注意力
        self.local_att1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att3 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        # self.w = nn.Parameter(torch.ones(4))

    def forward(self, y, x):
        xa = y + x
        xl1 = self.local_att1(xa)
        xl2 = self.local_att3(xa)
        xg1 = self.global_attmax(xa)
        xg2 = self.global_attavg(xa)

        xlg = xl1 + xl2 + xg1 + xg2

        wei = self.sigmoid(xlg)
        xi = x * wei + y

        return xi
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
class iAFF_our1(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF_our1, self).__init__()
        inter_channels = int(channels // r)
        # Global
        # 全局注意力
        self.global_attmax = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )
        self.global_attavg = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )

        # Local
        # 本地注意力
        self.local_att1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att3 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        # self.w = nn.Parameter(torch.ones(4))

    def forward(self, x, residual):
        xa = x + residual
        xl1 = self.local_att1(xa)
        xl2 = self.local_att3(xa)
        xg1 = self.global_attmax(xa)
        xg2 = self.global_attavg(xa)

        xlg = xl1 + xl2 + xg1 + xg2

        # xlg = xl1 + xl2
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        return xi
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
class iAFFmod(nn.Module):
    def __init__(self, channels=64, r=4):
        super(iAFFmod, self).__init__()
        inter_channels = int(channels // r)
        # Global全局注意力
        self.global_attmax = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )
        self.global_attavg = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1, bias=False)
        )

        # Local本地注意力
        self.local_att1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att3 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        # self.w = nn.Parameter(torch.ones(4))

    def forward(self, x, residual):
        xa = abs(x - residual)
        # xa = x + residual
        xl1 = self.local_att1(xa)
        xl2 = self.local_att3(xa)
        xg1 = self.global_attmax(xa)
        xg2 = self.global_attavg(xa)
        xlg = xl1 + xl2 + xg1 + xg2
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl11 = self.local_att1(xi)
        xl22 = self.local_att3(xi)
        xg11 = self.global_attmax(xi)
        xg22 = self.global_attavg(xi)

        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        # w4 = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))

        xlg2 = xl11 + xl22 + xg11 + xg22
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo
# ----------------------------------------------------------------------------------------------------------------------


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)


class convup(nn.Module):
    def __init__(self, out_ch):
        super(convup, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(512, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.Conv2d(1024, out_ch, kernel_size=1, stride=1, padding=0, bias=True),

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.up(x)
        return x



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
        self.linear_pred4 = nn.Sequential(Conv2d(embedding_dim512, embedding_dim64, kernel_size=1),
                                          Conv2d(embedding_dim64, self.num_classes, kernel_size=1))

        self.upconv3 = up_conv(512, 256)
        self.upconv2 = up_conv(256, 128)
        self.upconv1 = up_conv(128, 64)
        # self.upconv3 = up_conv(1024, 512)
        # self.upconv2 = up_conv(512, 256)
        # self.upconv1 = up_conv(256, 128)

        self.convcc3 = Conv2d(256, 256, kernel_size=1)
        self.convcc2 = Conv2d(128, 128, kernel_size=1)
        self.convcc1 = Conv2d(64, 64, kernel_size=1)
        self.conv = conv3(512, 512)
        # self.convcc3 = Conv2d(512, 512, kernel_size=1)
        # self.convcc2 = Conv2d(256, 256, kernel_size=1)
        # self.convcc1 = Conv2d(128, 128, kernel_size=1)
        # self.conv = conv3(1024, 1024)

        self.dropout = nn.Dropout(0.1)

        self.iaff1 = iAFF_method1(64)
        self.iaff2 = iAFF_method1(128)
        self.iaff3 = iAFF_method1(256)


    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        # c1 1x64x88x88   c2 1x128x44x44   c3 1x320x22x22   c4 1x512x11x11



        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c3 = self.upconv3(c4)  # 1x320x22x22
        # _c3 = torch.cat((_c3, c3), dim=1) # 1x640x22x22
        # _c3 = _c3 + c3
        _c3 = self.iaff3(_c3, c3)
        _c3 = self.convcc3(_c3)  # 1x320x22x22

        _c2 = self.upconv2(_c3)  # 1x128x44x44
        # _c2 = torch.cat((_c2, c2), dim=1) # 1X256X44X44
        # _c2 = _c2 + c2
        _c2 = self.iaff2(_c2, c2)
        _c2 = self.convcc2(_c2)  # 1x128x44x44

        _c1 = self.upconv1(_c2)  # 1X64X88X88
        # _c1 = torch.cat((_c1, c1), dim=1) # 1X128X88X88
        # _c1 = _c1 + c1
        _c1 = self.iaff1(_c1, c1)
        _c1 = self.convcc1(_c1)  # 1X64X88X88

        x = self.dropout(_c1)
        x = self.linear_pred(x)  # 1x1x88x88

        return x


@SEGMENTORS.register_module()
class colonformer_ssformer_method1(nn.Module):
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
        super(colonformer_ssformer_method1, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=[64, 128, 320, 512], class_num=1)
        # self.decode_head = Decoder(dims=[128, 256, 512, 1024], dim=[128, 256, 512, 1024], class_num=1)

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
        # x4 = segout[0]  #  1x64x88x88
        # x3 = segout[1]  # 1x128x44x44
        # x2 = segout[2]  # 1x320x22x22
        # x1 = segout[3]  # 1x512x11x11

        # Uper Decoder
        # predict out1
        decoder_1 = self.decode_head.forward([x1, x2, x3, x4])
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')  # 1x1x352x352
        # lateral_map_1 = F.interpolate(decoder_1, scale_factor=2, mode='bilinear')  # 1x1x352x352

        return lateral_map_1
        # return torch.sigmoid(lateral_map_1)