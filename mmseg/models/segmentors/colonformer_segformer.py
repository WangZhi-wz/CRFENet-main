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


class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        # self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
        #                           nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # 1x121x256 1x484x256 1x1936x256 1x7744x256
        x = self.proj(x)  # 1x256x11x11 1x256x22x22 1x256x44x44 1x256x88x88

        return x


class Decoder(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, dims, dim, class_num=2):
        super(Decoder, self).__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))

        self.linear_pred = Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])  # 1x512x11x11 -> 1x256x11x11
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)  # 1x256x88x88
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])  # 1x320x22x22 -> 1x256x22x22
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)  # 1x256x88x88
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])  # 1x128x44x44 -> 1x256x44x44
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)  # 1x256x88x88
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])  # 1x64x88x88 -> 1x256x88x88

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)  # 1x1x88x88

        return x


@SEGMENTORS.register_module()
class ColonFormer_segformer(nn.Module):
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
        super(ColonFormer_segformer, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=256, class_num=1)
        # ResNet
        # self.decode_head = Decoder(dims=[64, 128, 320, 512], dim=256, class_num=1)
        # UNet
        self.decode_head = Decoder(dims=[128, 256, 512, 1024], dim=128, class_num=1)

        # self.decode_head = builder.build_head(decode_head)
        # self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        # self.decode_head.init_weights()

        self.CFP_1 = CFPModule(128, d=8)
        self.CFP_2 = CFPModule(320, d=8)
        self.CFP_3 = CFPModule(512, d=8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(128, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra2_conv1 = Conv(320, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra3_conv1 = Conv(512, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.aa_kernel_1 = AA_kernel(128, 128)
        self.aa_kernel_2 = AA_kernel(320, 320)
        self.aa_kernel_3 = AA_kernel(512, 512)

    def forward(self, x):
        segout = self.backbone(x)
        # x1 = segout[0]  #  1x64x88x88
        # x2 = segout[1]  # 1x128x44x44
        # x3 = segout[2]  # 1x320x22x22
        # x4 = segout[3]  # 1x512x11x11
        x4 = segout[0]  #  1x64x88x88
        x3 = segout[1]  # 1x128x44x44
        x2 = segout[2]  # 1x320x22x22
        x1 = segout[3]  # 1x512x11x11

        # Uper Decoder
        # predict out1
        decoder_1 = self.decode_head.forward([x1, x2, x3, x4])  # 1x1x88x88
        # lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')  # 1x1x352x352
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=2, mode='bilinear')  # 1x1x352x352


        # return lateral_map_1
        return torch.sigmoid(lateral_map_1)

        # # ------------------- RA-RA attention-one -----------------------
        # decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear') # 1x1x11x11
        #
        # #12.10
        # cfp_out_1 = self.CFP_3(x4) # 1x512x11x11
        # # cfp_out_1 = x4
        #
        #
        # # cfp_out_1 += x4
        # # Reverse
        # decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1 # 1x1x11x11
        # # Residual axial attention
        # aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        # aa_atten_3 += cfp_out_1
        # # Reverse attention
        # aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(aa_atten_3) # 1x512x11x11
        #
        # # predictConv feature (C-> 1)
        # ra_3 = self.ra3_conv1(aa_atten_3_o)
        # ra_3 = self.ra3_conv2(ra_3)
        # ra_3 = self.ra3_conv3(ra_3) # 1x1x11x11
        #
        # # predict out2
        # x_3 = ra_3 + decoder_2  # 1x1x11x11
        # lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear') # 1x1x352x352
        #
        #
        #
        # # ------------------- RA-RA attention-two -----------------------
        # decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear') # 1x1x22x22
        #
        #
        # # 12.10
        # cfp_out_2 = self.CFP_2(x3) # 1x320x22x22
        # # cfp_out_2 = x3
        #
        #
        # # cfp_out_2 += x3
        # # Reverse
        # decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1 # 1x1x22x22
        # # Residual axial attention
        # aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        # aa_atten_2 += cfp_out_2
        # # Reverse attention
        # aa_atten_2_o = decoder_3_ra.expand(-1, 320, -1, -1).mul(aa_atten_2) # 1x320x22x22
        #
        # # predictConv feature (C-> 1)
        # ra_2 = self.ra2_conv1(aa_atten_2_o)
        # ra_2 = self.ra2_conv2(ra_2)
        # ra_2 = self.ra2_conv3(ra_2)
        #
        # # predict out3
        # x_2 = ra_2 + decoder_3 # 1x1x22x22
        # lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear') # 1x1x352x352
        #
        #
        #
        # # ------------------- RA-RA attention-three -----------------------
        # decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear') # 1x1x44x44
        #
        #
        # # 12.10
        # cfp_out_3 = self.CFP_1(x2) # 1x128x44x44
        # # cfp_out_3 = x2
        #
        # # cfp_out_3 += x2
        # # Reverse
        # decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1 # 1x1x44x44
        # # Residual axial attention
        # aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        # aa_atten_1 += cfp_out_3
        # # Reverse attention
        # aa_atten_1_o = decoder_4_ra.expand(-1, 128, -1, -1).mul(aa_atten_1) # 1x128x44x44
        #
        # # predictConv feature (C-> 1)
        # ra_1 = self.ra1_conv1(aa_atten_1_o)
        # ra_1 = self.ra1_conv2(ra_1)
        # ra_1 = self.ra1_conv3(ra_1)
        #
        # # predict out4
        # x_1 = ra_1 + decoder_4 # 1x1x44x44
        # lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') # 1x1x352x352
        #
        # return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1