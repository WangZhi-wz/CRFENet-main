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



# ----------------------------------------------------------------------------------------------------------------------
# Background Guide Module
class Spade(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(Spade, self).__init__()
        self.param_free_norm = nn.BatchNorm2d(out_channels, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, edge):
        # x: 1 64 88 88,   edge: 1 1 88 88
        normalized = self.param_free_norm(x) # 1 64 88 88

        edge = F.interpolate(edge, size=x.size()[2:], mode='nearest') # 1 1 88 88
        actv = self.mlp_shared(edge) # 1 64 88 88
        gamma = self.mlp_gamma(actv) # 1 64 88 88
        beta = self.mlp_beta(actv) # 1 64 88 88
        out = normalized * (1 + gamma) + beta # 1 64 88 88
        return out

    def initialize(self):
        weight_init(self)
# ----------------------------------------------------------------------------------------------------------------------



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
        self.dropout = nn.Dropout2d(0.1)
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

        self.convcc3 = Conv2d(320, 320, kernel_size=1)
        self.convcc2 = Conv2d(128, 128, kernel_size=1)
        self.convcc1 = Conv2d(64, 64, kernel_size=1)
        self.convcc3 = ConvBlock(320, 320, 3,1,1)
        self.convcc2 = ConvBlock(128, 128, 3,1,1)
        self.convcc1 = ConvBlock(64, 64, 3,1,1)

        self.conv = conv3(512, 512)

        self.dropout = nn.Dropout(0.1)

        # self.aff1 = AFF(64)
        # self.aff2 = AFF(128)
        # self.aff3 = AFF(320)
        # self.iaff1 = iAFF(64)
        # self.iaff2 = iAFF(128)
        # self.iaff3 = iAFF(320)
        self.iaff1 = iAFFmod(64)
        self.iaff2 = iAFFmod(128)
        self.iaff3 = iAFFmod(320)

        # self.RME1 = RF2B(64, 64)
        # self.RME2 = RF2B(128, 128)
        # self.RME3 = RF2B(320, 320)
        # self.RME4 = RF2B(512, 512)

        self.edge_conv1 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.edge_conv2 = BasicConv2d(128, 64, kernel_size=3, padding=1)
        self.edge_conv3 = BasicConv2d(320, 64, kernel_size=3, padding=1)
        self.edge_conv4 = BasicConv2d(512, 64, kernel_size=3, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.spade1 = Spade(64, 64)
        self.spade2 = Spade(128, 128)
        self.spade3 = Spade(320, 320)
        self.spade4 = Spade(512, 512)

        self.edge_conv_cat = BasicConv2d(64 * 4, 64, kernel_size=3, padding=1)
        self.edge_linear = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.sideout3 = SideoutBlock(512, 1)
        self.sideout2 = SideoutBlock(320, 1)
        self.sideout1 = SideoutBlock(128, 1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        # c1 1x64x88x88    c2 1x128x44x44    c3 1x320x22x22    c4 1x512x11x11

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        # # out1
        # # x1 = self.dropout(_c1)
        # c1_out = self.sideout3(c4)  # 1x1x88x88
        # # LCA
        # score1 = torch.sigmoid(c1_out)
        # dist1 = torch.abs(score1 - 0.5)
        # att1 = 1 - (dist1 / 0.5)


        # _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])   # 1x512x11x11 -> 1x512x11x11

        # c1, c2, c3, c4 = self.RME1(c1), self.RME2(c2), self.RME3(c3), self.RME4(c4)

        # Boundary Detector: 3x3Conv + BN
        c1_edge = self.edge_conv1(c1) # 1 64 88 88
        c2_edge = self.edge_conv2(c2) # 1 64 44 44
        c3_edge = self.edge_conv3(c3) # 1 64 22 22
        c4_edge = self.edge_conv4(c4) # 1 64 11 11

        c2_edge = self.upsample2(c2_edge) # 1 64 88 88
        c3_edge = self.upsample4(c3_edge) # 1 64 88 88
        c4_edge = self.upsample8(c4_edge) # 1 64 88 88

        edge_cat = self.edge_conv_cat(torch.cat((c1_edge, c2_edge, c3_edge, c4_edge), dim=1)) # 1 64 88 88
        # edge_cat = self.edge_conv_cat(torch.cat((c2_edge, c3_edge, c4_edge), dim=1))  # 1 64 88 88
        edge_map1 = self.edge_linear(edge_cat) # 1 1 88 88
        edge_map2 = F.interpolate(edge_map1, size=c2.size()[2:], mode='bilinear')
        edge_map3 = F.interpolate(edge_map1, size=c3.size()[2:], mode='bilinear')
        # edge_map4 = F.interpolate(edge_map1, size=c4.size()[2:], mode='bilinear')

        # c4 = self.spade4(c4, edge_map4) # 1 320 22 22


        _c3 = self.upconv3(c4) # 1x320x22x22
        # _c3 = torch.cat((_c3, c3), dim=1) # 1x640x22x22
        _c3 = self.spade3(_c3, edge_map3) # 1 320 22 22
        _c3 = self.iaff3(_c3, c3)
        _c3 = self.convcc3(_c3) # 1x320x22x22

        # # out2
        # # x1 = self.dropout(_c1)
        # c2_out = self.sideout2(_c3)  # 1x1x88x88
        # # LCA
        # score2 = torch.sigmoid(c2_out)
        # dist2 = torch.abs(score2 - 0.5)
        # att2 = 1 - (dist2 / 0.5)

        _c2 = self.upconv2(_c3) # 1x128x44x44
        # _c2 = torch.cat((_c2, c2), dim=1) # 1X256X44X44
        _c2 = self.spade2(_c2, edge_map2) # 1X256X44X44
        _c2 = self.iaff2(_c2, c2)
        _c2 = self.convcc2(_c2) # 1x128x44x44

        # # out3
        # # x1 = self.dropout(_c1)
        # c3_out = self.sideout1(_c2)  # 1x1x88x88
        # # LCA
        # score3 = torch.sigmoid(c3_out)
        # dist3 = torch.abs(score3 - 0.5)
        # att3 = 1 - (dist3 / 0.5)

        _c1 = self.upconv1(_c2) # 1X64X88X88
        # _c1 = torch.cat((_c1, c1), dim=1) # 1X128X88X88
        _c1 = self.spade1(_c1, edge_map1) # 1X128X88X88
        _c1 = self.iaff1(_c1, c1)
        _c1 = self.convcc1(_c1) # 1X64X88X88

        x = self.dropout(_c1)
        x = self.linear_pred(x)  # 1x1x88x88

        return x
        # return x,c3_out,c2_out,c1_out



@SEGMENTORS.register_module()
class colonformer_ssformer_0403_AFF_BG(nn.Module):
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
        super(colonformer_ssformer_0403_AFF_BG, self).__init__()
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

        # pred4, pred3, pred2, pred1 = self.decode_head.forward([x1, x2, x3, x4]) # 1x1x88x88
        # lateral_map_4 = F.interpolate(pred4, scale_factor=4, mode='bilinear') # 1x1x352x352
        # lateral_map_3 = F.interpolate(pred3, scale_factor=8, mode='bilinear') # 1x1x352x352
        # lateral_map_2 = F.interpolate(pred2, scale_factor=16, mode='bilinear')  # 1x1x352x352
        # lateral_map_1 = F.interpolate(pred1, scale_factor=32, mode='bilinear')  # 1x1x352x352
        # return lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1
