# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFP(nn.Module):
    def __init__(self, channel):
        super(AFP, self).__init__()

        self.branch1 = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),
        )

        self.branch2 = nn.Sequential(
            nn.AvgPool2d(3, 1, padding=1),
        )

        self.branch3_1 = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 3, 1, 1),
            nn.Conv2d(channel // 2, channel // 2, 1),
        )
        self.branch3_2 = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 1),
            nn.Conv2d(channel // 2, channel // 2, 1),
        )

        self.w = nn.Parameter(torch.ones(4))

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3_1 = self.branch3_1(x)
        b3_2 = self.branch3_2(x)
        b3 = torch.cat([b3_1, b3_2], dim=1)
        b4 = x

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        w4 = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))

        x_out = b1 * w1 + b2 * w2 + b3 * w3 + b4 * w4
        print("权重：", w1, w2, w3, w4)
        print("权重：", w1.tolist(), w2.tolist(), w3.tolist(), w4.tolist())

        return x_out


if __name__ == '__main__':
    model = AFP(512)
    print(model)
    inputs = torch.randn(1, 512, 8, 8)
    outputs = model(inputs)
