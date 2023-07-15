import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

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

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


@SEGMENTORS.register_module()
class ColonFormer_mod(nn.Module):
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
        super(ColonFormer_mod, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        
        self.CFP_1 = CFPModule(128, d = 8)
        self.CFP_2 = CFPModule(320, d = 8)
        self.CFP_3 = CFPModule(512, d = 8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(128,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(320,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(512,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = AA_kernel(128,128)
        self.aa_kernel_2 = AA_kernel(320,320)
        self.aa_kernel_3 = AA_kernel(512,512)

        # Decoder
        self.convcenter = Conv(512,512,3,1,padding=1,bn_acti=True)
        self.Up_conv4 = conv_block(1024, 512)
        self.decoder3 = DecoderBlock(in_channels=512, out_channels=320)
        self.Up_conv3 = conv_block(640, 320)
        self.decoder2 = DecoderBlock(in_channels=320, out_channels=128)
        self.Up_conv2 = conv_block(256, 128)
        self.decoder1 = DecoderBlock(in_channels=128, out_channels=64)
        self.Up_conv1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  #  1x64x88x88
        x2 = segout[1]  # 1x128x44x44
        x3 = segout[2]  # 1x320x22x22
        x4 = segout[3]  # 1x512x11x11


        #Decoder
        x44 = self.convcenter(x4)
        x44 = torch.cat([x44, x4], dim=1)
        x44 = self.Up_conv4(x44)
        # predictConv feature (C-> 1)
        ra_3 = self.ra3_conv1(x44)
        ra_3 = self.ra3_conv2(ra_3)
        ra_3 = self.ra3_conv3(ra_3) # 1x1x11x11
        lateral_map_4 = F.interpolate(ra_3,scale_factor=32,mode='bilinear') # 1x1x352x352

        x33 = self.decoder3(x44)
        x33 = torch.cat([x33, x3], dim=1)
        x33 = self.Up_conv3(x33)
        # predictConv feature (C-> 1)
        ra_2 = self.ra2_conv1(x33)
        ra_2 = self.ra2_conv2(ra_2)
        ra_2 = self.ra2_conv3(ra_2)
        lateral_map_3 = F.interpolate(ra_2,scale_factor=16,mode='bilinear') # 1x1x352x352

        x22 = self.decoder2(x33)
        x22 = torch.cat([x22, x2], dim=1)
        x22 = self.Up_conv2(x22)
        # predictConv feature (C-> 1)
        ra_1 = self.ra1_conv1(x22)
        ra_1 = self.ra1_conv2(ra_1)
        ra_1 = self.ra1_conv3(ra_1)
        lateral_map_2 = F.interpolate(ra_1,scale_factor=8,mode='bilinear') # 1x1x352x352

        x11 = self.decoder1(x22)
        x11 = torch.cat([x11, x1], dim=1)
        x11 = self.Up_conv1(x11)
        x11 = self.final(x11)
        lateral_map_1 = F.interpolate(x11, scale_factor=4, mode='bilinear') # 1x1x352x352

        return lateral_map_4,lateral_map_3,lateral_map_2,lateral_map_1


        # # Uper Decoder
        # # predict out1
        # decoder_1 = self.decode_head.forward([x1, x2, x3, x4]) # 1x1x88x88
        # lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear') # 1x1x352x352
        #
        #
        # # ------------------- RA-RA attention-one -----------------------
        # decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear') # 1x1x11x11
        # cfp_out_1 = self.CFP_3(x4) # 1x512x11x11
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
        # cfp_out_2 = self.CFP_2(x3) # 1x320x22x22
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
        # cfp_out_3 = self.CFP_1(x2) # 1x128x44x44
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