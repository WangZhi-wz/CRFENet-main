import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d, init
from mmseg.core import add_prefix
from mmseg.ops import resize
# from .odconv import ODConv2d
from .grid_attention_layer import MultiAttentionBlock
from .networks_other import init_weights
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


# Shuffle Grouped Fusion Module
# ----------------------------------------------------------------------------------------------------------------------

class Convs(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Convs, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class asa_layer(nn.Module):
    def __init__(self, channel, groups=16):
        super().__init__()
        self.groups = groups

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // groups, 1, 1))

        self.cbias = nn.Parameter(torch.ones(1, channel // groups, 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // groups, 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel //  groups, 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // groups, channel //  groups)
        self.oneconv1 = nn.Conv2d(channel*3, channel, 1)

        # bi-linear modelling for both
        self.W_g = Convs(channel, channel, 1, bn=True, relu=False)
        self.W_x = Convs(channel, channel, 1, bn=True, relu=False)
        self.W = Convs(channel, channel, 3, bn=True, relu=True)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x, y):
        b, c, h, w = x.shape  # x(1,64,192,192) y(1,64,192,192)

        # bilinear pooling
        W_g = self.W_g(x) # 1,64,88,88
        W_x = self.W_x(y) # 1,64,88,88
        bp = self.W(W_g * W_x) # 1,64,88,88
        bp = bp.reshape(b * self.groups, -1, h, w) # 16,4,88,88

        x = x.reshape(b * self.groups, -1, h, w) # 16,4,88,88
        y = y.reshape(b * self.groups, -1, h, w) # 16,4,88,88
        x_0 = x # 16,4,88,88
        x_1 = y # 16,4,88,88

        xn = self.avg_pool(x_0) # 16,4,1,1
        xn = self.cweight * xn + self.cbias # 16,4,1,1
        xn = x_0 * self.sigmoid(xn) # 16,4,88,88

        xs = self.gn(x_1) # 16,4,88,88
        xs = self.sweight * xs + self.sbias # 16,4,88,88
        xs = x_1 * self.sigmoid(xs) # 16,4,88,88

        # concatenate along channel axis
        out = torch.cat([xn, xs, bp], dim=1) #16,12,88,88
        out = out.reshape(b, -1, h, w) #1,192,88,88

        out = self.channel_shuffle(out, 3) #1,192,88,88
        out = self.oneconv1(out) #1,64,88,88
        return out

# ----------------------------------------------------------------------------------------------------------------------

class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm1 = norm_layer(out_dim)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.act(self.norm1(self.conv(x)))
        return x


""" Channel Attention Module"""

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


""" Spatial Attention Module"""

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class CSAMBlock(nn.Module):

    def __init__(self, channel, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel,reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x*self.ca(x)
        out = out*self.sa(out)
        return out+residual


""" Global Context Module"""

"""
Non Local Block

https://arxiv.org/abs/1711.07971
"""
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class GCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM, self).__init__()
        pool_size = [1, 3, 5]
        out_channel_list = [64]
        upsampe_scale = [8]
        GClist = []
        GCoutlist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(        # Global average pool -> 3x3AdaptiveAvgPool -> 5x5AdaptiveAvgPool
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True)))
        GClist.append(nn.Sequential(            # Non-Local
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True),
            NonLocalBlock(out_channels)))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(1):      # 0 1 2 3
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)

    def forward(self, x):
        xsize = x.size()[2:]
        global_context = []     # 0 1 2
        for i in range(len(self.GCmodule) - 1):     # 0 1 2
            global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
        global_context.append(self.GCmodule[-1](x))     # 0 1 2 3
        global_context = torch.cat(global_context, dim=1)

        output = []
        for i in range(len(self.GCoutmodel)):       # 0 1 2 3
            output.append(self.GCoutmodel[i](global_context))

        return output

# ASM module
class ASMcasa(nn.Module):
    def __init__(self, in_channels, rate = 4):
        super(ASMcasa, self).__init__()
        self.non_local = NonLocalBlock(in_channels)
        # self.convl = nn.Conv2d(in_channels,in_channels //2, kernel_size=1, stride=1)
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention(kernel_size=7)


    def forward(self, fuse, low, high):

        fuse1 = fuse + low
        fuse1 = self.ca(fuse1) * fuse1
        fuse2 = fuse + high
        fuse2 = self.sa(fuse2) * fuse2
        # fuse_all = torch.cat([fuse1, fuse2], dim=1)
        fuse_all = fuse1 + fuse2

        return fuse_all

# CA-Net----------------------------------------------------------------------------------------------------------------

# CA-Net----------------------------------------------------------------------------------------------------------------



class Decoder(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """


    def __init__(self, dims, dim, class_num=2, nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super(Decoder, self).__init__()
        self.num_classes = class_num

        embedding_dim64, embedding_dim128, embedding_dim320, embedding_dim512 = dim[0], dim[1], dim[2], dim[3]
        # dims = [64, 128, 320, 512], dim = [64, 128, 320, 512]

        self.outconv = nn.Sequential(
            nn.Conv2d(embedding_dim64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, self.num_classes, 1))

        self.dropout = nn.Dropout(0.1)

        self.sgfm1_12 = asa_layer(64)
        self.sgfm1_23 = asa_layer(128)
        self.sgfm1_34 = asa_layer(320)
        self.sgfm2_12 = asa_layer(64)
        self.sgfm2_23 = asa_layer(128)
        self.sgfm3_12 = asa_layer(64)

        # self.upsample12 = Upsample(128, 64)
        # self.upsample13 = Upsample(320, 128)
        # self.upsample14 = Upsample(512, 320)
        # self.upsample21 = Upsample(128, 64)
        # self.upsample22 = Upsample(320, 128)
        # self.upsample31 = Upsample(128, 64)

        self.cbam = CSAMBlock(channel=64)
        self.gcm = GCM(512, 128)

        self.asm = ASMcasa(64)

        self.linear_pred = Conv2d(embedding_dim64, self.num_classes, kernel_size=1)

        self.attentionblock12 = MultiAttentionBlock(in_size=64, gate_size=128, inter_size=64,
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock23 = MultiAttentionBlock(in_size=128, gate_size=320, inter_size=128,
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock34 = MultiAttentionBlock(in_size=320, gate_size=512, inter_size=320,
                                                 nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)


    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        # c1 1x64x88x88  c2 1x128x44x44  c3 1x256x22x22  c4 1x512x11x11


        c1_12 = self.attentionblock12(c1, c2)[0] # 1 64 88 88

        c1_23 = self.attentionblock23(c2, c3)[0] # 1 128 44 44

        c1_34 = self.attentionblock34(c3, c4)[0] # 1 320 22 22

        c2_12 = self.attentionblock12(c1_12, c1_23)[0]

        c2_23 = self.attentionblock23(c1_23, c1_34)[0]

        c3_12 = self.attentionblock12(c2_12, c2_23)[0]

        c1_cbam = self.cbam(c1)
        # c4 gcm enhanced
        c4_gcm = self.gcm(c4)[0]

        # fuse1
        cout = c3_12 + c1_cbam
        cout = cout + c4_gcm
        x = self.linear_pred(cout)  # 1x1x88x88

        # ASM
        # cout = self.asm(c3_12, c1_cbam, c4_gcm)
        # x = self.outconv(cout)  # 1x1x88x88

        return x



@SEGMENTORS.register_module()
class colonformer_ssformer_moddecoder_CA(nn.Module):
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
        super(colonformer_ssformer_moddecoder_CA, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=[64, 128, 320, 512], class_num=1)
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)

        
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