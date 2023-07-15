import torch
import torch.nn as nn
import torch.nn.functional as F

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

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn

@SEGMENTORS.register_module()
class ColonFormer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped 0during inference.
    """

    def __init__(self,
                 backbone,  # backbones -> mix_transformer
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(ColonFormer, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        # self.align_corners = self.decode_head.align_corners
        # self.num_classes = self.decode_head.num_classes

        self.backbone.init_weights(pretrained=pretrained)
        # self.decode_head.init_weights()
        
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

        self.spatial_gating_unit1 = AttentionModule(128)
        self.spatial_gating_unit2 = AttentionModule(320)
        self.spatial_gating_unit3 = AttentionModule(512)

        
    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  #  1x64x88x88
        x2 = segout[1]  # 1x128x44x44
        x3 = segout[2]  # 1x320x22x22
        x4 = segout[3]  # 1x512x11x11

        # Uper Decoder
        # predict out1
        decoder_1 = self.decode_head.forward([x1, x2, x3, x4]) # 1x1x88x88
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear') # 1x1x352x352


        # ------------------- RA-RA attention-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear') # 1x1x11x11

        #12.10
        cfp_out_1 = self.CFP_3(x4) # 1x512x11x11
        # cfp_out_1 = x4
        # cfp_out_1 = self.spatial_gating_unit3(x4)

        # Reverse
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1 # 1x1x11x11
        # Residual axial attention
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3 += cfp_out_1
        # Reverse attention
        aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(aa_atten_3) # 1x512x11x11
        
        # predictConv feature (C-> 1)
        ra_3 = self.ra3_conv1(aa_atten_3_o)
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) # 1x1x11x11

        # predict out2
        x_3 = ra_3 + decoder_2  # 1x1x11x11
        lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear') # 1x1x352x352



        # ------------------- RA-RA attention-two -----------------------
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear') # 1x1x22x22

        # 12.10
        cfp_out_2 = self.CFP_2(x3) # 1x320x22x22
        # cfp_out_2 = x3
        # cfp_out_2 = self.spatial_gating_unit2(x3)

        # Reverse
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1 # 1x1x22x22
        # Residual axial attention
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2 += cfp_out_2
        # Reverse attention
        aa_atten_2_o = decoder_3_ra.expand(-1, 320, -1, -1).mul(aa_atten_2) # 1x320x22x22

        # predictConv feature (C-> 1)
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2)

        # predict out3
        x_2 = ra_2 + decoder_3 # 1x1x22x22
        lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear') # 1x1x352x352



        # ------------------- RA-RA attention-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear') # 1x1x44x44

        # 12.10
        cfp_out_3 = self.CFP_1(x2) # 1x128x44x44
        # cfp_out_3 = x2
        # cfp_out_3 = self.spatial_gating_unit1(x2)

        # Reverse
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1 # 1x1x44x44
        # Residual axial attention
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1 += cfp_out_3
        # Reverse attention
        aa_atten_1_o = decoder_4_ra.expand(-1, 128, -1, -1).mul(aa_atten_1) # 1x128x44x44

        # predictConv feature (C-> 1)
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1)

        # predict out4
        x_1 = ra_1 + decoder_4 # 1x1x44x44
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') # 1x1x352x352
        
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1