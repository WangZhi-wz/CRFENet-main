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


# DW Conv   DW-D-Conv  1x1Conv
# 添加---------------------------------------------------------------------------------
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
# 添加---------------------------------------------------------------------------------


# 添加---------------------------------------------------------------------------------
""" Channel Attention Module"""

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
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

    def __init__(self, channel=512, reduction=16, kernel_size=49):
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
# 添加---------------------------------------------------------------------------------


# BLOCKS to construct the model
# 添加---------------------------------------------------------------------------------
class DSDF_block(nn.Module):  #OK
    def __init__(self, in_ch_x, in_ch_y, nf1=128, nf2=256, gc=64, bias=True):
        super().__init__()

        self.nx1 = nn.Sequential(nn.Conv2d(in_ch_x, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

        self.ny1 = nn.Sequential(nn.Conv2d(in_ch_y, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))
        # 缩小尺寸
        self.nx1c = nn.Sequential(nn.Conv2d(in_ch_x, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),    # ks 3 -> 4, stride 1 -> 2
                                  nn.LeakyReLU(negative_slope=0.25))
        # 增加尺寸
        self.ny1t = nn.Sequential(nn.ConvTranspose2d(in_ch_y, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),   # ks 3 -> 4
                                  nn.LeakyReLU(negative_slope=0.25))

        self.nx2 = nn.Sequential(nn.Conv2d(in_ch_x+gc+gc, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                  nn.LeakyReLU(negative_slope=0.25))

        self.ny2 = nn.Sequential(nn.Conv2d(in_ch_y+gc+gc, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                  nn.LeakyReLU(negative_slope=0.25))

        self.nx2c = nn.Sequential(nn.Conv2d(gc, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),    # ks 3 -> 4, stride 1 -> 2
                                  nn.LeakyReLU(negative_slope=0.25))

        self.ny2t = nn.Sequential(nn.ConvTranspose2d(gc, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),   # ks 3 -> 4
                                  nn.LeakyReLU(negative_slope=0.25))

        self.nx3 = nn.Sequential(nn.Conv2d(in_ch_x+gc+gc+gc, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

        self.ny3 = nn.Sequential(nn.Conv2d(in_ch_y+gc+gc+gc, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

        self.nx3c = nn.Sequential(nn.Conv2d(gc, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),    # ks 3 -> 4, stride 1 -> 2
                                  nn.LeakyReLU(negative_slope=0.25))

        self.ny3t = nn.Sequential(nn.ConvTranspose2d(gc, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),   # ks 3 -> 4
                                  nn.LeakyReLU(negative_slope=0.25))

        self.nx4 = nn.Sequential(nn.Conv2d(in_ch_x+gc+gc+gc+gc, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

        self.ny4 = nn.Sequential(nn.Conv2d(in_ch_y+gc+gc+gc+gc, gc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

        self.nx4c = nn.Sequential(nn.Conv2d(gc, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),    # ks 3 -> 4, stride 1 -> 2
                                  nn.LeakyReLU(negative_slope=0.25))

        self.ny4t = nn.Sequential(nn.ConvTranspose2d(gc, gc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=bias),   # ks 3 -> 4
                                  nn.LeakyReLU(negative_slope=0.25))

        self.nx5 = nn.Sequential(nn.Conv2d(in_ch_x+gc+gc+gc+gc+gc, nf1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

        self.ny5 = nn.Sequential(nn.Conv2d(in_ch_y+gc+gc+gc+gc+gc, nf2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=bias),
                                 nn.LeakyReLU(negative_slope=0.25))

    def forward(self, x, y):
        # 1x64x88x88  1x128x44x44   1x320x22x22   1x512x11x11
        x1 = self.nx1(x) # 1x32x88x88   1x128x22x22
        y1 = self.ny1(y) # 1x32x44x44   1x128x11x11

        x1c = self.nx1c(x) # 1x32x44x44   1x128x11x11
        y1t = self.ny1t(y) # 1x32x88x88   1x128x22x22

        x2_input = torch.cat([x, x1, y1t], dim=1) # 1x128x88x88   1x576x22x22
        x2 = self.nx2(x2_input) # 1x32x88x88   1x128x22x22

        y2_input = torch.cat([y, y1, x1c], dim=1) # 1x192x44x44   1x768x11x11
        y2 = self.ny2(y2_input) # 1x32x44x44   1x128x11x11

        x2c = self.nx2c(x1) # 1x32x44x44    1x128x11x11
        y2t = self.ny2t(y1) # 1x32x88x88    1x128x22x22

        x3_input = torch.cat([x, x1, x2, y2t], dim=1) # 1x160x88x88   1x704x22x22
        x3 = self.nx3(x3_input) # 1x32x88x88    1x128x22x22

        y3_input = torch.cat([y, y1, y2, x2c], dim=1) # 1x224x44x44   1x896x11x11
        y3 = self.ny3(y3_input) # 1x32x44x44    1x128x11x11

        x3c = self.nx3c(x3) # 1x32x44x44 1x128x11x11
        y3t = self.ny3t(y3) # 1x32x88x88 1x128x22x22

        x4_input = torch.cat([x, x1, x2, x3, y3t], dim=1) # 1x192x88x88   1x832x22x22
        x4 = self.nx4(x4_input) # 1x32x88x88   1x128x22x22

        y4_input = torch.cat([y, y1, y2, y3, x3c], dim=1)  # 1x256x44x44   1x1024x11x11
        y4 = self.ny4(y4_input) # 1x32x44x44   1x128x11x11

        x4c = self.nx4c(x4) # 1x32x44x44   1x128x11x11
        y4t = self.ny4t(y4) # 1x32x88x88   1x128x22x22

        x5_input = torch.cat([x, x1, x2, x3, x4, y4t], dim=1) # 1x224x88x88   1x960x22x22
        x5 = self.nx5(x5_input) # 1x64x88x88   1x320x22x22

        y5_input = torch.cat([y, y1, y2, y3, y4, x4c], dim=1) # 1x288x44x44   1x1152x11x11
        y5 = self.ny5(y5_input) # 1x128x44x44   1x512x11x11

        x5 *= 0.4 # 1x64x88x88   1x320x22x22
        y5 *= 0.4 # 1x128x44x44  1x512x11x11

        return x5+x, y5+y
# 添加---------------------------------------------------------------------------------



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

        self.csam1 = CSAMBlock(64)
        self.csam2 = CSAMBlock(128)
        self.csam3 = CSAMBlock(320)
        self.csam4 = CSAMBlock(512)

        # one stage
        self.dsfs_1 = DSDF_block(init_feat, init_feat * 2, nf1=init_feat, nf2=init_feat * 2, gc=init_feat // 2)
        self.dsfs_2 = DSDF_block(init_feat*5, init_feat*8, nf1=init_feat*5, nf2=init_feat*8, gc=init_feat*4//2)
        # two stage
        self.dsfs_3 = DSDF_block(init_feat, init_feat*2, nf1=init_feat, nf2=init_feat*2, gc=init_feat//2)
        self.dsfs_4 = DSDF_block(init_feat*5, init_feat*8, nf1=init_feat*5, nf2=init_feat*8, gc=init_feat*4//2)
        # three stage fusion
        self.dsfs_5  = DSDF_block(init_feat*2, init_feat*5, nf1=init_feat*2, nf2=init_feat*5, gc=init_feat*2//2)
        # four stage
        self.dsfs_6  = DSDF_block(init_feat, init_feat*2, nf1=init_feat, nf2=init_feat*2, gc=init_feat//2)
        self.dsfs_7  = DSDF_block(init_feat*5, init_feat*8, nf1=init_feat*5, nf2=init_feat*8, gc=init_feat*4//2)
        # five stage fusion
        self.dsfs_8  = DSDF_block(init_feat*2, init_feat*5, nf1=init_feat*2, nf2=init_feat*5, gc=init_feat*2//2)
        # six stage
        self.dsfs_9  = DSDF_block(init_feat, init_feat*2, nf1=init_feat, nf2=init_feat*2, gc=init_feat//2)
        self.dsfs_10 = DSDF_block(init_feat*5, init_feat*8, nf1=init_feat*5, nf2=init_feat*8, gc=init_feat*4//2)

        self.convdowm1 = nn.Conv2d(c1_in_channels, c1_in_channels//2, 1)
        self.convdowm2 = nn.Conv2d(c2_in_channels, c2_in_channels//2, 1)
        self.convdowm3 = nn.Conv2d(c3_in_channels, c3_in_channels//2, 1)
        self.convdowm4 = nn.Conv2d(c4_in_channels, c4_in_channels//2, 1)
        self.convup1 = nn.Conv2d(c1_in_channels // 2, c1_in_channels , 1)
        self.convup2 = nn.Conv2d(c2_in_channels // 2, c2_in_channels, 1)
        self.convup3 = nn.Conv2d(c3_in_channels // 2, c3_in_channels, 1)
        self.convup4 = nn.Conv2d(c4_in_channels // 2, c4_in_channels, 1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        # c1 1x64x88x88
        # c2 1x128x44x44
        # c3 1x320x22x22
        # c4 1x512x11x11


        # # MSDF add
        # c1 = self.convdowm1(c1)
        # c2 = self.convdowm2(c2)
        # c3 = self.convdowm3(c3)
        # c4 = self.convdowm4(c4)
        #
        # # one stage
        # x12, x22 = self.dsfs_1(c1, c2)
        # x32, x42 = self.dsfs_2(c3, c4)
        # # two stage
        # x12, x22 = self.dsfs_3(x12, x22)
        # x32, x42 = self.dsfs_4(x32, x42)
        # # three stage fusion
        # x22, x32 = self.dsfs_5(x22, x32)
        # # four stage
        # x13, x23 = self.dsfs_6(x12, x22)
        # x33, x43 = self.dsfs_7(x32, x42)
        # # five stage fusion
        # x23, x33 = self.dsfs_8(x23, x33)
        # # six stage
        # x13, x23 = self.dsfs_9(x13, x23)
        # x33, x43 = self.dsfs_10(x33, x43)
        #
        # c1 = (x13*0.4) + c1
        # c2 = (x23*0.4) + c2
        # c3 = (x33*0.4) + c3
        # c4 = (x43*0.4) + c4
        #
        # c1 = self.convup1(c1)
        # c2 = self.convup2(c2)
        # c3 = self.convup3(c3)
        # c4 = self.convup4(c4)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])  # 1x512x11x11 -> 1x512x11x11

        _c3 = resize(_c4, size=c3.size()[2:], mode='bilinear', align_corners=False)  # 1x512x22x22
        _c3 = self.linear_c3(_c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])  # 1x320x22x22
        # c3attention = self.spatial_gating_unit3(c3)
        # c3attention = self.csam3(c3)
        _c3 = self.linear_fuse3(torch.cat([_c3, c3], dim=1))  # 1x320x22x22
        # x3 = self.linear_pred3(_c3)

        _c2 = resize(_c3, size=c2.size()[2:], mode='bilinear', align_corners=False)  # 1x320x44x44
        _c2 = self.linear_c2(_c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])  # 1x128x44x44
        # c2attention = self.spatial_gating_unit2(c2)
        # c2attention = self.csam2(c2)
        _c2 = self.linear_fuse2(torch.cat([_c2, c2], dim=1))  # 1x320x22x22
        # x2 = self.linear_pred2(_c2)


        _c1 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)  # 1x128x88x88
        _c1 = self.linear_c1(_c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])  # 1x64x88x88
        # c1attention = self.spatial_gating_unit1(c1)
        # c1attention = self.csam1(c1)
        _c1 = self.linear_fuse1(torch.cat([_c1, c1], dim=1))  # 1x64x88x88

        x = self.dropout(_c1)
        x = self.linear_pred(x)  # 1x1x88x88

        # return x, x2, x3
        return x


@SEGMENTORS.register_module()
class colonformer_mitbaseline(nn.Module):
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
        super(colonformer_mitbaseline, self).__init__()
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

        self.out1 = Conv2d(512, 64, kernel_size=1)
        self.out2 = Conv2d(64, 1, kernel_size=1)

        
    def forward(self, x):
        segout = self.backbone(x)
        x1 = segout[0]  #  1x64x88x88
        x2 = segout[1]  # 1x128x44x44
        x3 = segout[2]  # 1x320x22x22
        x4 = segout[3]  # 1x512x11x11

        # Uper Decoder
        # predict out1
        # decoder_1, decoder_2, decoder_3 = self.decode_head.forward([x1, x2, x3, x4]) # 1x1x88x88 1x1x44x44 1x1x22x22
        # decoder_1 = self.decode_head.forward([x1, x2, x3, x4])
        # lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear') # 1x1x352x352
        # lateral_map_2 = F.interpolate(decoder_2, scale_factor=8, mode='bilinear')  # 1x1x352x352
        # lateral_map_3 = F.interpolate(decoder_3, scale_factor=16, mode='bilinear')  # 1x1x352x352

        x4 = self.out1(x4)
        x4 = self.out2(x4)
        lateral_map_1 = F.interpolate(x4, scale_factor=32, mode='bilinear') # 1x1x352x352


        # return lateral_map_1, lateral_map_2, lateral_map_3
        return lateral_map_1
        # return torch.sigmoid(lateral_map_1)