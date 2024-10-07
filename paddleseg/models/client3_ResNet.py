# -*- coding: utf-8 -*-

import paddle.nn as nn
# 导入两个必要的包

from paddleseg.utils import utils
import paddle.nn.functional as F
from paddleseg.cvlibs import manager
from paddle.vision.ops import DeformConv2D as DCN


def swish(x):
    return F.relu(x)


class residualBlock(nn.Layer):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2D(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2D(n)
        self.conv2 = nn.Conv2D(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2D(n)

        self.relu = nn.ReLU()

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = y + x
        return self.relu(y)


@manager.MODELS.add_component
class Generator_sar_client3(nn.Layer):
    def __init__(self, n_residual_blocks=6, pretrained=None, num_classes=2):
        super(Generator_sar_client3, self).__init__()
        self.pretrained = pretrained
        self.n_residual_blocks = n_residual_blocks
        self.conv1 = nn.Conv2D(1, 64, 9, stride=1, padding=4)

        self.resblock = self.make_layer(residualBlock, self.n_residual_blocks)
        # for i in range(self.n_residual_blocks):

        # self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2D(64, 64, 3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2D(64)
        self.bn2 = nn.BatchNorm2D(64)
        self.offsets = nn.Conv2D(64, 18, kernel_size=3, stride=1, padding=1)
        self.dconv2_2 = DCN(64, 64, 3, 1, 1)
        self.dconv2_3 = DCN(64, 64, 3, 1, 1)
        self.dconv2_4 = DCN(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2D(64, 2, 1, stride=1, padding=0)
        self.drop = nn.Dropout2D(0.4)
        self.init_weight()

    def make_layer(self, block, num_layers):
        res_block = []
        for i in range(num_layers):
            res_block.append(block())
        return nn.Sequential(*res_block)

    def forward(self, x):
        #########################original version########################
        x = self.conv1(x)

        y = x.clone()
        # for i in range(self.n_residual_blocks):
        #     y = self.resblock[i](y)

        y = self.resblock(y)
        x = self.bn2(self.conv2(y)) + x
        x = swish(x)
        x = self.drop(x)
        offset = self.offsets(x)
        x = self.dconv2_2(x, offset)
        x = self.dconv2_3(x, offset)
        # x = self.dconv2_4(x,offset)
        output = self.conv4(x)
        # print(x.shape)
        # return x
        return [output]
        # return x,self.conv4(x)

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
