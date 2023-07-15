import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d, init
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

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim512)
        self.linear_c3 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim320)
        self.linear_c2 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim128)
        self.linear_c1 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim64)
        #
        self.linear_fuse4 = ConvModule(in_channels=embedding_dim512 * 2, out_channels=embedding_dim512, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse3 = ConvModule(in_channels=embedding_dim320 * 2, out_channels=embedding_dim320, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        # self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
        #                                 norm_cfg=dict(type='BN', requires_grad=True))
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
        self.dropout = nn.Dropout(0.1)

        self.assf_fusion4 = ASFF_ddw(level=0)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        # c1 1x64x88x88
        # c2 1x128x44x44
        # c3 1x320x22x22
        # c4 1x512x11x11

        fused = self.assf_fusion4(c4, c3, c2, c1)  # MSR-EFM [1, 256, 28, 40]


        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = fused.shape

        _c4 = self.linear_c4(fused).permute(0, 2, 1).reshape(n, -1, fused.shape[2], fused.shape[3])  # 1x512x11x11 -> 1x512x11x11
        _c4 = self.linear_fuse4(torch.cat([_c4, c4], dim=1))

        _c3 = resize(_c4, size=c3.size()[2:], mode='bilinear', align_corners=False)  # 1x512x22x22
        _c3 = self.linear_c3(_c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])  # 1x320x22x22
        _c3 = self.linear_fuse3(torch.cat([_c3, c3], dim=1))  # 1x320x22x22
        # x3 = self.linear_pred3(_c3)

        _c2 = resize(_c3, size=c2.size()[2:], mode='bilinear', align_corners=False)  # 1x320x44x44
        _c2 = self.linear_c2(_c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])  # 1x128x44x44
        _c2 = self.linear_fuse2(torch.cat([_c2, c2], dim=1))  # 1x320x22x22
        # x2 = self.linear_pred2(_c2)


        _c1 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)  # 1x128x88x88
        _c1 = self.linear_c1(_c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])  # 1x64x88x88
        _c1 = self.linear_fuse1(torch.cat([_c1, c1], dim=1))  # 1x64x88x88

        x = self.dropout(_c1)
        x = self.linear_pred(x)  # 1x1x88x88

        # return x, x2, x3
        return x

# MS-DFM
def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage

class ASFF_ddw(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF_ddw, self).__init__()
        self.level = level
        self.dim = [512, 320, 128, 64]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(320, self.inter_dim, 3, 2)
            self.stride_level_2 = nn.Sequential(
                add_conv(128, self.inter_dim, 3, 2),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.stride_level_3 = nn.Sequential(
                add_conv(64, 128, 3, 2),
                add_conv(128, self.inter_dim, 3, 2),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.expand = add_conv(self.inter_dim, 512, 3, 1)  # ????


        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2, x_level_3):
        # x_level_0 1x512x11x11 x_level_1 1x320x22x22 x_level_2 1x128x44x44 x_level_3 1x64x88x88
        if self.level == 0:
            level_0_resized = x_level_0  # 1x512x11x11
            level_1_resized = self.stride_level_1(x_level_1)  # [1, 256, 28, 40]
            level_2_resized = self.stride_level_2(x_level_2)  # [1, 256, 28, 40]
            level_3_resized = self.stride_level_3(x_level_3)  # [1, 256, 28, 40]

        level_0_weight_v = self.weight_level_0(level_0_resized)  # [1, 16, 28, 40]
        level_1_weight_v = self.weight_level_1(level_1_resized)  # [1, 16, 28, 40]
        level_2_weight_v = self.weight_level_2(level_2_resized)  # [1, 16, 28, 40]
        level_3_weight_v = self.weight_level_3(level_3_resized)  # [1, 16, 28, 40]

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v),
                                    1)  # [1, 16, 28, 40]
        levels_weight = self.weight_levels(levels_weight_v)  # [1, 4, 28, 40]
        levels_weight = F.softmax(levels_weight, dim=1)  # [1, 4, 28, 40]

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:3, :, :] + \
                            level_3_resized * levels_weight[:, 3:, :, :]
        # [1, 256, 28, 40]
        out = self.expand(fused_out_reduced)  # [1, 256, 28, 40]

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


@SEGMENTORS.register_module()
class colonformer_ssformer_vanmsrdm(nn.Module):
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
        super(colonformer_ssformer_vanmsrdm, self).__init__()
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
        
        self.CFP_1 = CFPModule(128, d = 8)
        self.CFP_2 = CFPModule(320, d = 8)
        self.CFP_3 = CFPModule(512, d = 8)

    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  #  1x64x88x88
        x2 = segout[1]  # 1x128x44x44
        x3 = segout[2]  # 1x320x22x22
        x4 = segout[3]  # 1x512x11x11


        # Uper Decoder
        # predict out1
        # decoder_1, decoder_2, decoder_3 = self.decode_head.forward([x1, x2, x3, x4]) # 1x1x88x88 1x1x44x44 1x1x22x22
        decoder_1 = self.decode_head.forward([x1, x2, x3, x4])
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear') # 1x1x352x352
        # lateral_map_2 = F.interpolate(decoder_2, scale_factor=8, mode='bilinear')  # 1x1x352x352
        # lateral_map_3 = F.interpolate(decoder_3, scale_factor=16, mode='bilinear')  # 1x1x352x352

        # return lateral_map_1, lateral_map_2, lateral_map_3
        return lateral_map_1