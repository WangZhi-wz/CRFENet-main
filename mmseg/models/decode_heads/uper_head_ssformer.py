import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn import Conv2d

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM

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


@HEADS.register_module()
class UPerHead_ssformer(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, dims, dim, class_num=1):
        super(UPerHead_ssformer, self).__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
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

        L34 = self.linear_fuse34(torch.cat([_c4, _c3], dim=1))  # 1x256x88x88
        L2 = self.linear_fuse2(torch.cat([L34, _c2], dim=1))  # 1x256x88x88
        _c = self.linear_fuse1(torch.cat([L2, _c1], dim=1))  # 1x256x88x88

        x = self.dropout(_c)
        x = self.linear_pred(x)  # 1x1x88x88

        return x
