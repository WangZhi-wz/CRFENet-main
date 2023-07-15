import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
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
        x = self.proj(x) # 1x256x11x11 1x256x22x22 1x256x44x44 1x256x88x88
        x = x.flatten(2).transpose(1, 2) # 1x121x256 1x484x256 1x1936x256 1x7744x256
        return x


class conv1x1(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv1x1, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.up(x)
        return x

class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,strides=None,padding=0,ceil_mode = False,count_include_pad = True,divisor_override = None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,strides,padding,ceil_mode,count_include_pad,divisor_override)
    def forward(self, x):
        x_exp = torch.exp(x)    # 转为指数形式 1 32 224 320
        x_exp_pool = self.avgpool(x_exp)    # 1 32 112 160
        x = self.avgpool(x_exp*x)   # 1 32 112 160
        return x/x_exp_pool


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


# iAFFmod Module(Global:Avgpool + Maxpool, Local:1x1Conv + 3x3Conv)
# ----------------------------------------------------------------------------------------------------------------------
class iAFFmod(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFFmod, self).__init__()
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

        xl11 = self.local_att1(xi)
        xl22 = self.local_att3(xi)
        xg11 = self.global_attmax(xi)
        xg22 = self.global_attavg(xi)

        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        # w4 = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))

        # xlg2 = xl11*w1 + xl22*w2 + xg11*w3 + xg22*w4
        # print("权重：", w1.tolist(), w2.tolist(), w3.tolist(), w4.tolist())
        # xlg2 = xg11 + xg22
        xlg2 = xl11 + xl22 + xg11 + xg22
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# RF2B: RMFE module in paper.
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def initialize(self):
        weight_init(self)

# RF2B: RMFE module in paper.
class RF2B(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RF2B, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.branch4 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.conv = nn.Conv2d(in_channel, out_channel, 1)

        self.conv_cat = nn.Conv2d(out_channel*4, out_channel, 3, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(self.conv(x) + x1)
        x3 = self.branch3(self.conv(x) + x2)
        x4 = self.branch4(self.conv(x) + x3)
        x_cat = self.conv_cat(torch.cat((x1, x2, x3, x4), dim=1))

        x = self.relu(x0 + x_cat)
        return x

    def initialize(self):
        weight_init(self)

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is None:
                pass
            elif m.bias is not None:
                nn.init.zeros_(m.bias)
            else:
                nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.Upsample, Parameter, nn.AdaptiveAvgPool2d, nn.Sigmoid)):
            pass
        else:
            m.initialize()
# ----------------------------------------------------------------------------------------------------------------------


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):   # reduction8不如16
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


""" Local Context Attention Module"""

class LCA(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7):
        super(LCA, self).__init__()
        self.channel = in_channel

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True))

        self.ca = ChannelAttention(channel=in_channel,reduction=reduction)

    def forward(self, x, pred):
        residual = x
        pred = torch.sigmoid(pred)

        # boundary
        dist = torch.abs(pred - 0.5)
        att_boundary = 1 - (dist / 0.5)

        # foreground
        att_foreground = pred
        # att_foreground = torch.clip(att_foreground - att_boundary, 0, 1)
        x_foreground = att_foreground * x

        # background
        att_background = 1 - pred
        # att_background = torch.clip(att_background - att_boundary, 0, 1)
        x_background = att_background * x

        x_mid = self.conv(x_foreground + x_background)
        # x_mid = self.conv(torch.cat((x_foreground, x_background), dim=1))

        x_all = att_boundary * x_mid
        # x_all = self.conv1(x_all) + residual



        x_all = self.ca(x_all) + residual

        return x_all

def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = nn.BatchNorm2d
    normalization_layer = layer(in_channels)
    return normalization_layer


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)

class DPmod(nn.Module):
    def __init__(self, inplane, norm_layer):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(DPmod, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, stride=1, padding=1),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=1),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(inplane*2, inplane, kernel_size=3, stride=1, padding=1),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
        )

        self.flow_make = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x, edge, body):
        size = x.size()[2:]  # 1 128 44 44
        seg_down = self.down(x)  # 1 128 10 10
        seg_down = F.upsample(seg_down, size=size, mode="bilinear", align_corners=True)  # 1 128 44 44
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))  # 1 2 44 44
        seg_flow_warp = self.flow_warp(x, flow, size)  # 1 128 44 44

        # body
        seg_body = seg_flow_warp
        seg_body = seg_body + body
        seg_body = self.conv11(seg_body)
        # edge
        seg_edge = x - seg_flow_warp
        seg_edge = seg_edge + edge
        seg_edge = self.conv11(seg_edge)

        seg_all = self.conv(seg_body + seg_edge)
        seg_all = self.outconv(torch.cat((seg_all, x), dim=1))

        return seg_all

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device) # 1 1 1 2
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)  # 44 44
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)  # 44 44
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)  # 44 44 2

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)  # 1 44 44 2
        grid = grid + flow.permute(0, 2, 3, 1) / norm  # 1 44 44 2

        output = F.grid_sample(input, grid)  # 1 128 44 44
        return output


class DP(nn.Module):
    def __init__(self, inplane, norm_layer):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(DP, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, stride=1, padding=1),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=1),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(inplane*2, inplane, kernel_size=3, stride=1, padding=1),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
        )

        self.flow_make = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]  # 1 128 44 44
        seg_down = self.down(x)  # 1 128 10 10
        seg_down = F.upsample(seg_down, size=size, mode="bilinear", align_corners=True)  # 1 128 44 44
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))  # 1 2 44 44
        # body
        seg_flow_warp = self.flow_warp(x, flow, size)  # 1 128 44 44
        # edge
        seg_edge = x - seg_flow_warp
        seg_edge = seg_edge
        seg_edge = self.conv11(seg_edge)

        seg_all = self.conv(seg_flow_warp + seg_edge)
        seg_all = self.outconv(torch.cat((seg_all, x), dim=1))

        return seg_all

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device) # 1 1 1 2
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)  # 44 44
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)  # 44 44
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)  # 44 44 2

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)  # 1 44 44 2
        grid = grid + flow.permute(0, 2, 3, 1) / norm  # 1 44 44 2

        output = F.grid_sample(input, grid)  # 1 128 44 44
        return output


class Decoder(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """


    def __init__(self, dims, dim, class_num=2):
        super(Decoder, self).__init__()
        self.num_classes = class_num
        init_feat = 32
        channels = 128

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
        self.linear_pred = Conv2d(embedding_dim128, self.num_classes, kernel_size=1)
        self.linear_pred2 = nn.Sequential(Conv2d(embedding_dim128, embedding_dim64, kernel_size=1),
                                          Conv2d(embedding_dim64, self.num_classes, kernel_size=1))
        self.linear_pred3 = nn.Sequential(Conv2d(embedding_dim320, embedding_dim64, kernel_size=1),
                                          Conv2d(embedding_dim64, self.num_classes, kernel_size=1))
        self.linear_pred4 = nn.Sequential(Conv2d(embedding_dim512, embedding_dim64, kernel_size=1),
                                          Conv2d(embedding_dim64, self.num_classes, kernel_size=1))

        self.upconv3 = up_conv(128, 128)
        self.upconv2 = up_conv(128, 128)
        self.upconv1 = up_conv(128, 128)

        # self.convcc3 = Conv2d(128, 128, kernel_size=1)
        # self.convcc2 = Conv2d(128, 128, kernel_size=1)
        # self.convcc1 = Conv2d(128, 128, kernel_size=1)

        self.conv = conv3(512, 512)

        self.dropout = nn.Dropout(0.1)

        # self.aff1 = AFF(64)
        # self.aff2 = AFF(128)
        # self.aff3 = AFF(320)
        # self.iaff1 = iAFF(64)
        # self.iaff2 = iAFF(128)
        # self.iaff3 = iAFF(320)
        self.iaff1 = iAFFmod(128)
        self.iaff2 = iAFFmod(128)
        self.iaff3 = iAFFmod(128)

        # self.RME1 = RF2B(64, 64)
        # self.RME2 = RF2B(128, 128)
        # self.RME3 = RF2B(320, 320)
        # self.RME4 = RF2B(512, 512)

        self.lca4 = LCA(512)
        self.lca3 = LCA(320)
        self.lca2 = LCA(128)
        self.lca1 = LCA(64)

        # self.dp4 = DP(128, Norm2d)
        # self.dp3 = DP(128, Norm2d)
        # self.dp2 = DP(128, Norm2d)
        # self.dp1 = DP(128, Norm2d)
        self.dp4 = DPmod(128, Norm2d)
        self.dp3 = DPmod(128, Norm2d)
        self.dp2 = DPmod(128, Norm2d)
        self.dp1 = DPmod(128, Norm2d)

        self.side_conv1 = nn.Conv2d(512, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2 = nn.Conv2d(320, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3 = nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4 = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)

        self.softpool2 = conv1x1(128, 128)
        self.softpool3 = SoftPooling2D(2, 2)
        self.softpool4 = SoftPooling2D(4, 4)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        c1, c2, c3, c4 = self.side_conv4(c1), self.side_conv3(c2), self.side_conv2(c3), self.side_conv1(c4)
        # c1 1x128x88x88   c2 1x128x44x44   c3 1x128x22x22   c4 1x128x11x11

        # c4_out = self.linear_pred4(c4)
        # c3_out = self.linear_pred3(c3)
        # c2_out = self.linear_pred2(c2)
        # c1_out = self.linear_pred(c1)

        # c4 = self.lca4(c4, c4_out)
        # c3 = self.lca3(c3, c3_out)
        # c2 = self.lca2(c2, c2_out)
        # c1 = self.lca1(c1, c1_out)

        global_3 = F.interpolate(c4, size=c3.size()[2:], mode='bilinear') # 1x128x22x22
        global_2 = F.interpolate(c4, size=c2.size()[2:], mode='bilinear') # 1x128x44x44
        global_1 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear') # 1x128x88x88

        edge_2 = self.softpool2(c1) # 1x128x44x44
        edge_3 = self.softpool3(edge_2) # 1x128x22x22
        edge_4 = self.softpool4(edge_2) # 1x128x11x11

        c4 = self.dp4(c4, edge_4, c4)
        c3 = self.dp3(c3, edge_3, global_3)
        c2 = self.dp2(c2, edge_2, global_2)
        c1 = self.dp1(c1, c1, global_1)

        # c4 = self.dp4(c4)
        # c3 = self.dp3(c3)
        # c2 = self.dp2(c2)
        # c1 = self.dp1(c1)


        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape


        # _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])   # 1x512x11x11 -> 1x512x11x11

        # c1, c2, c3, c4 = self.RME1(c1), self.RME2(c2), self.RME3(c3), self.RME4(c4)



        _c3 = self.upconv3(c4) # 1x320x22x22
        # _c3 = torch.cat((_c3, c3), dim=1) # 1x640x22x22
        _c3 = self.iaff3(_c3, c3)
        # _c3 = self.convcc3(_c3) # 1x320x22x22

        _c2 = self.upconv2(_c3) # 1x128x44x44
        # _c2 = torch.cat((_c2, c2), dim=1) # 1X256X44X44
        _c2 = self.iaff2(_c2, c2)
        # _c2 = self.convcc2(_c2) # 1x128x44x44

        _c1 = self.upconv1(_c2) # 1X64X88X88
        # _c1 = torch.cat((_c1, c1), dim=1) # 1X128X88X88
        _c1 = self.iaff1(_c1, c1)
        # _c1 = self.convcc1(_c1) # 1X64X88X88

        x = self.dropout(_c1)
        x = self.linear_pred(x)  # 1x1x88x88

        return x



@SEGMENTORS.register_module()
class colonformer_ssformer_baseline0301_AFF_DPmod(nn.Module):
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
        super(colonformer_ssformer_baseline0301_AFF_DPmod, self).__init__()
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